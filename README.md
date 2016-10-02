[Slides](<https://docs.google.com/presentation/d/1w8tlsPJ5ISYcu4zeTcWbimKW5Amcif4duLac96DlzPo/embed?start=false&loop=false&delayms=3000)

[github] (https://github.com/priya-venkat)

[linkedin] (https://linkedin.com/in/priya26)


Customer Service is a huge part of the success equation of any business, and small companies lose out on this game because they do not have the human resources to provide adequate customer service.

So, when the company I was consulting for, whom we'll call Z-Star, came to me and said we have a user manual and an FAQ, but nobody ever reads the manual, I decided to help them to automate their customer service.

So what does this system look like?

User asks a question, the system looks for the question in the manual, retrieves the answer and gives it to the user. But of course, this would never work because Natural language is complicated and there are dozens of ways of asking the same question.

So, I needed to build a sort of Association Model that can take a customer question, extract relevant phrases and look for equivalent concepts in the repository of answers created from the company's data.

This is a challenging problem, and there's no off-the-shelf solution for this problem. 

A quick study of user questions revealed that on an average, questions were composed of a relation phrase, an entity phrase and can be answered with another entity phrase. 

I did some research and decided to use the Paraphrase driven learning model to help me with my task. 

First, I had to build a Dictionary of relations and entities that generalizes well to syntactic and lexical variations in the English language. 

I started with a set of paraphrased question pairs from Wikianswers. I used Natural Language Processing (CoreNLP) to tokenize and tag each word with a part of speech label. Then, I used string matching and boostrapped a seed vocabulary to extract equivalent relations and entities. 

One thing to note here is that the wikianswers corpus is crowdsourced, so there's a lot of noise which can lead to incorrect relations being learned. To filter out phrases that are less likely to be equivalent, I trained a multilayer perceptron network that assigns higher scores to phrases that are more likely to be equivalent. (like these)


In order to test this model, I used the Dialogue learning dataset from Facebook AI Research (FAIR) bAbI project.

As you can see, the Association model was able to extract the relation and entity phrases and match it against closely related concepts.

So, in conclusion, I was able to use the Association Model to automate customer questions answering at Z-Star, and as compared to exact string matching, my model increased the percentage of questions answered from 10% to 42%.



