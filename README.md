# CS6263 Faithful Chain of Thought Blog

### By Jason Johnson

## What is Faithful Chain of Thought?

Great Question!  Over the past few years Large Language Models have exploded on to the scene with amazing results in sentiment analysis, text completion, question answering, and more.  However, one area where LLMs have been lacking is in complex reasoning tasks, such as math word problems and common-sense reasoning.

Enter: Chain of Thought.  In 2022, Jason Wei and team introduced the concept in their paper, [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://openreview.net/forum?id=_VjQlMeSB_J).  They showed that if you ask the model to show their reasoning steps (its “Chain of Thought”), it significantly improves the ability of the model to perform those complex reasoning tasks.

Problem solved, right? Wrong!  Let’s not welcome our robot overlords just yet.  Look at this example:
 ![Picture1](https://github.com/jasonjay86/CS6263FaithfulCOT/assets/65077765/a9f322d6-95d6-40f2-87b2-0c091a843191)


Here the model is giving a nice, plausible looking Chain of Thought(CoT), but look closely…Its totally wrong.  Its gibberish.  Not only that, but it also gives the answer as 0.  But nowhere in the blue highlighted section will you find reasoning to give you a 0.  Lastly, if you ask the smartest 3rd grader you know, they will tell you the answer is 2 and not 0.  That’s 3 strikes on this output!

The problem is that the CoT can lie about the reasoning process.  It’s a form of hallucination for LLMs.  This can be very dangerous, no one wants to listen to a very plausible but very wrong explanation especially when it comes to their wallet, their health, or their freedom.

This is where Faithful COT comes in.  Veronica Lyu and her team introduced [Faithful Chain-of-Thought Reasoning](https://arxiv.org/pdf/2301.13379). A framework to make sure LLMs are giving their ture reasoning steps to derive their  answers.


## OK, So How does it Work?

## Does this work with other models?

## Adjusting to other models

## Results with Mistral

