# CS6263 Faithful Chain of Thought Blog

### By Jason Johnson

## What is Faithful Chain of Thought?

Great Question!  Over the past few years Large Language Models have exploded on to the scene with amazing results in sentiment analysis, text completion, question answering, and more.  However, one area where LLMs have been lacking is in complex reasoning tasks, such as math word problems and common-sense reasoning.

Enter: Chain of Thought.  In 2022, Jason Wei and team introduced the concept in their paper, [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://openreview.net/forum?id=_VjQlMeSB_J).  They showed that if you ask the model to show their reasoning steps (its “Chain of Thought”), it significantly improves the ability of the model to perform those complex reasoning tasks.

Problem solved, right? Wrong!  Let’s not welcome our robot overlords just yet.  Look at this example:
 ![Picture1](https://github.com/jasonjay86/CS6263FaithfulCOT/assets/65077765/a9f322d6-95d6-40f2-87b2-0c091a843191)


Here the model is giving a nice, plausible looking Chain of Thought(CoT), but look closely…Its totally wrong.  Its gibberish.  Not only that, but it also gives the answer as 0.  But nowhere in the blue highlighted section will you find reasoning to give you a 0.  Lastly, if you ask the smartest 3rd grader you know, they will tell you the answer is 2 and not 0.  That’s 3 strikes on this output!

The problem is that the CoT can lie about the reasoning process.  It’s a form of hallucination for LLMs.  This can be very dangerous, no one wants to listen to a very plausible but very wrong explanation especially when it comes to their wallet, their health, or their freedom.

This is where Faithful COT comes in.  Veronica Lyu and her team introduced [Faithful Chain-of-Thought Reasoning](https://arxiv.org/pdf/2301.13379). A framework to make sure LLMs are giving their true reasoning steps to derive their  answers.


## OK, So How does it Work?

I knew you would ask that!  Faithful Chain of Thought uses a two-stage pipeline.  With some few shot prompting, Faithful Chain of Thought forces the LLM to give its answer in a very specific way.   If you don’t know, few shot prompting means you provide the model with a few examples of your question-and-answer pair to help it provide you with a specific output format. One shot prompting would mean you give it just one example, zero shot is no examples.  For Faithful COT, they have prompted the model to output a reasoning chain that is a mix of natural language and symbolic language.  The natural language decomposes the question into multiple subproblems.  The symbolic language is a piece of code designed by the model to tackle each subproblem.  That mix of natural and symbolic language is called the Translation stage.  

The second stage they call the Problem-Solving stage.  The Translation from stage 1 is fed to a deterministic solver like a Python interpreter to calculate the answer **faithfully**.  No more 195-160 = 0, like was before!  In the paper and in their code, four types of problems were tested with four types of deterministic solvers.  For this blog post, I will focus on Math Word Problems, which used a Python interpreter for problem-solving.  Here’s an example of one of those problems:
![Picture2](https://github.com/jasonjay86/CS6263FaithfulCOT/assets/65077765/c4659aae-19ff-4777-836d-9ad7041ea3b1)

 
With this method, the model is forced to be faithful to the reasoning chain because the Python interpreter will literally follow the steps word for word.  However, it is important to note that the reasoning chain itself can still be wrong.  It can be *quite* wrong, in fact.

That said, the researchers produced excellent results across the board when applying this method.  They found that using this method with OpenAI's models made them more accurate than standard prompting or other forms of Chain of Thought prompting.
 ![Picture3](https://github.com/jasonjay86/CS6263FaithfulCOT/assets/65077765/7c3abe86-f6e9-4397-b732-d66dec0d6b22)

## Does this work with other models?

In reading the paper, I was a little surprised that the team only stuck with OpenAI’s LLMs.  Sure, OpenAI has arguably been the gold standard for LLMs for several years.  They certainly have the most famous LLMs in ChatGPT.  But they are not the only game in town!

I wondered *how well this technique could be used with a smaller model?*  So when we talk about model size, we are generally speaking of how many trainable parameters it has.  Without getting too math-y, we can say that the parameters are a rough indicator of a model’s ability to recognize complex patterns in data.  It’s *almost* an indication of how smart a model is (not really, but it helps). GPT-4, OpenAI’s latest and greatest, was trained on 1.76 *trillion* parameters.  Its predecessor, GPT-3, has 175 billion parameters.  Both of which were used in experiments on Faithful COT.  How does this work if you don’t have all those parameters to back you up?

To test this, I used an open-source LLM called [Mistral-7B](https://mistral.ai/technology/#models).  The 7B part means that it has just 7 billion parameters.  Still plenty to work with, right?  We will see.  Mistral is a series of models developed by a French based AI company, [Mistral AI](https://mistral.ai/company/).  They publish an instruction tuned version of the Mistral on [Hugging Face]( https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), available for anyone to use for free.  In case you don't know, [Hugging Face](https://huggingface.co/) is an incredible website full of machine learning and natural language processing models, datasets, and more for open collaboration in this space.  "Instruction tuned" means that the model has already been trained to take instructions.  Pretrained models that are not instruction tuned tend to just want to predict the next words it thinks you will say.

Armed with Mistral and the [code published by the Faithful COT team](https://github.com/veronica320/Faithful-COT), I set out to adapt Faithful COT to Mistral.

## Adjusting to other models

The logical first step for me was just getting the published code running for me.  Our friends at Penn on the Faithful COT team left us solid instructions in their README as well as an environment.yml file to build a Conda virtual environment.  After some typical python environment wrestling (I never have any luck with environment.yml files, even my own), I got to a point where I could run their `predict.py` program without errors.  `predict.py` is their main program that sends the math word problem with the “Faithful” few shot prompt to the model to get an output or a prediction.  So I run…

`python source/predict/predict.py --model_name gpt-3.5-turbo_standard --dataset_name SVAMP --split test`

to run the predict code on gpt 3.5 on the SVAMP dataset.  The SVAMP dataset is a set of elementary school level math word problems.  I picked that dataset because I wanted to do a math dataset and that one is easy to spell.

Now the published code is extremely well written and designed with plenty of comments and documentation, but for the life of me -- **I could not get this code to run.**  I could not understand it.  I tried changing my OpenAI key, I tried different models, added dozens of `print` statements, and nothing worked.  Finally, after hours of banging my head against my keyboard, I found this little line was the culprit:

`response = openai.ChatCompletion.create(…)`

That is a VERY important line, because it is the method that makes the call to OpenAI’s api to use their models.  To make a long story short, I believe openai changed their method name to:

`response = openai.chat.completions.create(…)`

With that simple change, everything worked perfectly.  I even saw results that mirrored what was published in the paper.  The next task was to make it work with Mistral instead of the OpenAI models. 

Without getting too technical, the code is implemented in such a way that the model is a custom python object, so the task boils down to making that class support Mistral.  I won’t get into every line of code I added to do that, but I’ll hit some highlights.  The code is in this repository if you are curious.

The model class is defined in the file `source/model/codex.py`.  In that class, I spent the most time in a method called `_query`.  In that method, the calls to the openai api are made.  There was already an if-then structure for the different versions of the api call, so I added to it for Mistral:

```
	elif LM in ["mistral"]: 
		completions = []
		device = "cuda" # the device to load the model onto
		model = AutoModelForCausalLM.from_pretrained(self.path)
		tokenizer = AutoTokenizer.from_pretrained(self.path)
		messages=[{"role": "user", "content": prompt}]
			
		encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
		model_inputs = encodeds.to(device)
		model.to(device)

		generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
		response = tokenizer.batch_decode(generated_ids)
		choices = response
   			
		for choice in choices:
			completions.append(self.get_Mistral_answer(choice))
```

That bit of code loads the model and tokenizer from Hugging Face, tokenizes the prompt and sends the tokenized prompt to the model through `model.generate()`.  It then decodes the response and splits it up into a format the original Faithful COT code likes.  Most of that code is directly from the [Mistral Hugging Face page]( https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), but adapted to work here.  I also needed to write an extra function called `get_Mistral_answer()` to format Mistral’s response to fit the rest of the code.

And that’s basically it.  Mistral, locked and loaded.  Let’s run it!

## Results with Mistral

Running through the test set of the SVAMP dataset goes through 1000 word problems, so it takes a good bit of time to execute.  In the interest of time, I stuck with the SVAMP dataset and ran three problem solving techniques.
* **Standard prompting** where we ask the model for the answer with no reasoning chain
* **Chain of Thought** where we ask the model for a reasoning chain along with an answer
* **Faithful COT** where we ask the model for a reasoning chain that can be run through a deterministic solver.  In this case, a python interpreter.

Here are the Mistral results (the numbers are the percentage of questions answered correctly):

|	| **SVAMP** |
|-----------------------------|:---------:|
| **Standard** |   46.6    |
| **COT**  |   46.2    |
| **Faithful COT**     | **58.5**  | 

As promised, Faithful COT improved Mistral’s test scores by 25%.  Good job, Mistral!  Its not exactly a passing grade, but improvement is good.

Interestingly the standard chain of thought run did slightly worse than the standard prompting.  I suspect it has to do with the 168 invalid answers that Mistral gave in the COT run.  Sometimes Mistral tried to be a little too smart for its own good. For this  question:

```
The Razorback shop makes $ 76 dollars off each jersey and $ 204 off each t-shirt. During the Arkansas and Texas tech game
they sold 158 t-shirts and 2 jerseys. How much money did they make from selling the jerseys?
```
COT Mistral gave me this:
```
The Razorback shop made $204 off each t-shirt and sold 158 of them, so they made 158 * $204 = $31,572 from the t-shirts.
They made $76 off each jersey and sold 2 of them, so they made 2 * $76 = $152 from the jerseys.
Therefore, they made a total of $31,572 (from t-shirts) + $152 (from jerseys) = $31,724.

However, the provided information stated that they made $31,572 off of the t-shirts and jerseys combined. To satisfy the
information given, the price of each jersey must have been underestimated. The correct number of jerseys sold and the
price per jersey should be adjusted accordingly to match the given total.

Let's assume that each jersey was sold for $60. Now we can calculate the number of jerseys sold: $152 (revenue from
jerseys) / $60 (price per jersey) = 2.53. Since it's not possible to sell a fraction of a jersey, we must round down
the number of jerseys sold to 2.

So, they sold 2 jerseys.
Therefore, the Razorback shop sold 158 t-shirts and 2 jerseys and made a total of $31,572 (from t-shirts) + $128 (from jerseys) = $31,698.
```
It tried to answer a lot more questions than what it was asked and ended veering off a cliff.  So the Faithful COT code scored it `[INVALID]`

Faithful COT Mistral gives us this elegant code:

```
# 1. How much money does the Razorback shop make from selling one jersey? (independent, support: [\"The Razorback shop makes $ 76 dollars off each jersey\"])
money_per_jersey = 76

# 2. How much money does the Razorback shop make from selling one t-shirt? (independent, support: [\"The Razorback shop makes $ 204 off each t-shirt\"])
money_per_t_shirt = 204

# 3. How many t-shirts were sold during the Arkansas and Texas tech game? (independent, support: [\"158 t-shirts were sold\"])
t_shirts_sold = 158

# 4. How many jerseys were sold during the Arkansas and Texas tech game? (independent, support: [\"2 jerseys were sold\"])
jerseys_sold = 2

# 5. How much money did the Razorback shop make from selling the jerseys? (depends on 1 and 3, support: [])
money_jerseys = money_per_jersey * jerseys_sold

# 6. Final Answer: How much money did they make from selling the jerseys? (depends on 5, support: [])
answer = money_jerseys
```

Which gives the correct answer, faithfully.

Finally, let's compare Mistral-Faithful COT with other models(from the paper):

|	| **SVAMP** |
|-----------------------------|:---------:|
| **Mistral Faithful COT** |   58.5    |
| **Codex**  |   83.5    |
| **ChatGPT**     | 83.0  | 
| **GPT-4**     | **95.3**  | 

So Mistral was the worst by far.  However we did show a significant improvement on Mistral over other promptimg styles.

## Wrapping It Up

While Mistral-7B didn't quite match the reasoning prowess of the big boys like GPT-4, the Faithful Chain of Thought approach definitely gave the little guy a boost. By breaking problems down into bite-sized pieces and solving them step-by-step, Faithful CoT helped Mistral improve its score on those tricky math word problems by over 25%. Not too shabby!

Sure, the larger language models still came out on top overall. But being able to enhance reasoning abilities in smaller, more resource-friendly models is a promising sign. As AI continues evolving, techniques like Faithful CoT could help democratize advanced reasoning capabilities across a wider range of applications.

So while Mistral may not be ushering in our robot overlord future just yet, this experiment shows there's still plenty of room for the smaller models to keep leveling up their reasoning game. The faithful revolution may be a little way off, but it's definitely one to watch.
