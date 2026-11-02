#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import argparse
import json
import os

import time

NUM_SECONDS_TO_SLEEP = 2
MAX_RETRIES = 8

# Judge model: defaults to the one used by the original LLaVA-Bench / BharatBench numbers.
# Override with --judge-model or the JUDGE_MODEL env var (e.g. "gpt-4o") if it is unavailable,
# but note that scores from different judges are NOT directly comparable.
DEFAULT_JUDGE = os.getenv("JUDGE_MODEL", "gpt-4-0613")

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("OPENAI_API_KEY is not set - the GPT judge cannot run.")

try:  # openai>=1.0
    from openai import OpenAI
    _client = OpenAI()

    def _chat(model, messages, temperature, max_tokens):
        r = _client.chat.completions.create(model=model, messages=messages,
                                            temperature=temperature, max_tokens=max_tokens)
        return r.choices[0].message.content
except ImportError:  # legacy openai==0.28
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def _chat(model, messages, temperature, max_tokens):
        r = openai.ChatCompletion.create(model=model, messages=messages,
                                         temperature=temperature, max_tokens=max_tokens)
        return r['choices'][0]['message']['content']


def get_eval(content: str, max_tokens: int, model: str = DEFAULT_JUDGE):
    messages = [{
        'role': 'system',
        'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
    }, {
        'role': 'user',
        'content': content,
    }]
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return _chat(model, messages, temperature=0.2, max_tokens=max_tokens)
        except Exception as e:  # rate limits, transient network errors, ...
            last_err = e
            print(f"[judge] attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            time.sleep(NUM_SECONDS_TO_SLEEP * (2 ** attempt))
    raise RuntimeError(f"GPT judge failed after {MAX_RETRIES} attempts: {last_err}")


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    parser.add_argument('--judge-model', type=str, default=DEFAULT_JUDGE, help='OpenAI model used as the judge')
    args = parser.parse_args()

    n_q = sum(1 for _ in open(os.path.expanduser(args.question), encoding='utf-8'))
    n_a = sum(1 for _ in open(os.path.expanduser(args.answer_list[1]), encoding='utf-8'))
    if n_a != n_q:
        raise SystemExit(f"{args.answer_list[1]} has {n_a} answers but {args.question} has {n_q} questions. "
                         "Answers are matched to questions line-by-line, so every question must be answered, in order.")

    f_q = open(os.path.expanduser(args.question), encoding='utf-8')
    f_ans1 = open(os.path.expanduser(args.answer_list[0]), encoding='utf-8')
    f_ans2 = open(os.path.expanduser(args.answer_list[1]), encoding='utf-8')
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output), encoding='utf-8')]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a', encoding='utf-8')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context), encoding='utf-8')]
    image_to_context = {context['image']: context for context in context_list}

    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        category = 'llava_bench_' + json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': category
        }
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens, args.judge_model)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            cur_js['judge'] = args.judge_model
            review_file.write(json.dumps(cur_js, ensure_ascii=False) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()
