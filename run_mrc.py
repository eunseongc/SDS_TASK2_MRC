import os
import torch
import argparse
import logging

from time import time
from tqdm import tqdm

from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig, BertForQuestionAnswering, squad_convert_examples_to_features, get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadV1Processor
from transformers.data.metrics.squad_metrics import squad_evaluate

from tokenization_kobert import KoBertTokenizer
from utils import set_seed, SquadResult, compute_predictions_logits
from evaluate import eval_during_train

logger = logging.getLogger(__name__)
EVAL_METRICS=["exact", "f1", "total"]

def load_and_cache_examples(args, tokenizer, mode=None, output_examples=False):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode,
            args.model_name.split("/")[-1],
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        processor = SquadV1Processor()
        if mode == 'dev':
            examples = processor.get_dev_examples(args.data_dir, filename=args.dev_file)
        elif mode == 'test':
            examples = processor.get_dev_examples(args.data_dir, filename=args.test_file)
        else:
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)


        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training = mode=='train',
            return_dataset="pt",
            threads=args.threads,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    # Added here for reproductibility
    set_seed(args.seed)

    for _ in tqdm(range(int(args.num_train_epochs)), desc="Epoch", dynamic_ncols=True):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            
            outputs = model(**inputs)

            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

        # Log metrics, Evaluate & save model checkpoint
        if args.eval_during_training:
            logger.info("***** Eval results *****")
            results = evaluate(args, model, tokenizer)

            for key in EVAL_METRICS:
                logger.info("  %s = %s", key, str(results[key]))

            # Write the evaluation result on file
            logger.info("***** Syllable unit evaluation results *****")
            with open("eval_result_{}_{}.txt".format(list(filter(None, args.model_name.split("/"))).pop(),
                                                        str(args.max_seq_length)), "a", encoding='utf-8') as f:
                official_eval_results = eval_during_train(args)
                f.write(f"Step: {global_step}\n")
                for key in sorted(official_eval_results.keys()):
                    logger.info("  %s = %s", key, str(official_eval_results[key]))
                    f.write(f" {key} = {str(official_eval_results[key])}\n")

        print("lr", scheduler.get_lr()[0], global_step)
        print("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
        logging_loss = tr_loss

        output_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)
        
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, mode="dev", prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, mode=mode, output_examples=True)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_dataloader = DataLoader(dataset, shuffle=False, batch_size=args.eval_batch_size)

    # Eval!
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = time()

    for batch in tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]
            outputs = model(**inputs)
            outputs = list(outputs.values())

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [output[i].detach().cpu().tolist() for output in outputs]
            start_logits, end_logits = output

            result = SquadResult(unique_id, start_logits, end_logits)
            all_results.append(result)

    evalTime = time() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, f"predictions_{mode}_{prefix}.json")
    output_nbest_file = os.path.join(args.output_dir, f"nbest_predictions_{mode}_{prefix}.json")

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        tokenizer
    )

    results = squad_evaluate(examples, predictions)

    return results

def main(args):
    set_seed(args.seed)

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train):
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty.")

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S",level=logging.INFO)

    # Load pretrained model and tokenizer
    config = BertConfig.from_pretrained(args.model_name)
    tokenizer = KoBertTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    model = BertForQuestionAnswering.from_pretrained(args.model_name, from_tf=bool(".ckpt" in args.model_name), config=config)

    model.to(args.device)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode='train', output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        print("######### Training is done #########")
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    results = {}
    if args.do_eval:
        logger.info("Loading checkpoints saved during training for evaluation")
        checkpoint_path = os.path.join(args.output_dir, args.checkpoint_dir)
        
        global_step = checkpoint_path.split("-")[-1] if len(checkpoint_path) > 0 else ""
        logger.info("Evaluate the following checkpoints: %s", checkpoint_path)

        # Reload the model
        model = BertForQuestionAnswering.from_pretrained(checkpoint_path)
        model.to(args.device)

        # Evaluate
        result = evaluate(args, model, tokenizer, mode='test', prefix=global_step)

        result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
        results.update(result)

    logger.info("Results: {}".format(results))

    return results


# python run_mrc.py --model_name monologg/kobert --output_dir outputs/kobert --eval_during_training --do_train
# python run_mrc.py --model_name monologg/kobert --output_dir outputs/kobert --checkpoint_dir checkpoint-XXX --do_eval

def parse_args():
    parser = argparse.ArgumentParser()

    # These arguments must be specified.
    parser.add_argument("--model_name", default=None, type=str, required=True, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.")

    # Other parameters
    parser.add_argument("--checkpoint_dir", default=None, type=str, required=False, help="Path to the checkpoint directory")
    parser.add_argument("--data_dir", default="data/mrc", type=str, help="The path of dataset")
    parser.add_argument("--train_file", default="news_mrc_train.json", type=str, help="The file name of training dataset")
    parser.add_argument("--dev_file", default="news_mrc_dev.json", type=str, help="The file name of validation dataset")
    parser.add_argument("--test_file", default="news_mrc_test.json", type=str, help="The file name of test dataset")


    parser.add_argument("--max_seq_length", default=512, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int, help="When splitting up a long document into chunks, how much stride to take between chunks.")
    
    parser.add_argument("--max_query_length", default=64, type=int, help="The maximum number of tokens for the question. Questions longer than this will be truncated to this length.")
   
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")    
    parser.add_argument("--eval_during_training", action="store_true", help="Run evaluation during training at each logging step.")

    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")


    parser.add_argument("--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int, help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")

    parser.add_argument("--logging_steps", type=int, default=4000, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=4000, help="Save checkpoint every X updates steps.")

    parser.add_argument("--seed", type=int, default=85453, help="random seed for initialization")
    parser.add_argument("--threads", type=int, default=8, help="multiple threads for converting example to features")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = parse_args()
    main(args)

