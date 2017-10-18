import os
from src.utils.vocab_utils import lookup_vocab
from src.utils.data_utils import iobes_iob

eval_path = "./src/metrics"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")


def test_ner(results, global_step):
    """
    Run perl script to evaluate model
    """
    output_path = os.path.join(eval_path, "eval.%i.output" % global_step)
    scores_path = os.path.join(eval_path, "eval.%i.scores" % global_step)

    with open(output_path, "w") as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)

    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))
    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in open(scores_path, 'r', encoding='utf8')]

    return eval_lines


def external_evaluate(sess, model, global_step, words_id_to_vocab, tags_id_to_vocab, iterator):
    results = []
    for batched_data in iterator.minibatches():
        predicted_ids = model.decode(sess, batched_data)
        sources, seq_lens, _, _, targets = batched_data
        for word_ids, lable_line, pred_line, length in zip(sources, targets, predicted_ids, seq_lens):
            assert len(pred_line) == length, "predict length is error"
            words = lookup_vocab(words_id_to_vocab, word_ids[:length])
            lables = lookup_vocab(tags_id_to_vocab, lable_line[:length])
            preds = lookup_vocab(tags_id_to_vocab, pred_line[:length])
            result = []
            for char, label, pred in zip(words, lables, preds):
                result.append(" ".join([char, label, pred]))
            results.append(result)

    return results
