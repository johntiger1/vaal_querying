# May 5
After running a huge search/experiment, come back with mostly null results. This is not necessarily good or bad, but could likely mean 
too much training of the policy NN

too much noise. (noise set to 0.81)

We did notice that it was able to avoid collapse, thankfully.

Either way, no big deal. We can still try the large batch sampling, and see what happens. But time is a limited resource these days...


# May 4
The goal right now is the following.
1. we have shown that *some* learning may occur.
2. we have also shown that batching is unaddressed by uncertainty.

Therefore, we foresee a use case for the RL if we can produce K outputs, and then we reshape. We can do both a sequential, as well as a batch version. 

That is: in one approach, we produce N\*K outputs, and then reshape, and train. In the other one, we simply retrain sequentially, stepping the network each time...

Less principled, but...

We will *specifically* show that our RL network solves the batched problem, and ONLY the batched problem

Interesting, we can simply sample N times, *from* the parameters that our NN outputs! This can work!

However, this probably won't work. The reason is that, it will just learn to concentrate all the probs on one area, again! 
If instead, we have it *predicting the context*, then we will probably have concentration again. 

Overall, the state representation and such is too challenging for this problem. 

We have never been able to solve the concetration problem yet.

OK, so I want action concentration then: 


Ultimately, these are the **strong** assumptions we make. 
1. we assume that we know the prior distribution of the testing set (not too bad)
2. we assume that there is some signficant correlation between k-means clustering and real labels. If this is not the case, then, we can also use a noisy labeller. 

Under these two assumptions, therefore, we would like to show that RL is useful
and particularly, that RL can be improved from a naive by using the prior penalty.

So naturally, the idea is to a) show that RL can be useful over time (done). 2) show that on average, giving a roughly average distribution of points is more useful than not. 

This assumes that our data distribution decision boundary is roughly even, which it is not. 

What about: support points identified as support vectors by the SVM?? (even though this is more work, it will probably be better)

I know! Hacking into the Kmeans, and essentially finding the boundary points! The uncertainty points *for* the kmeans. Instead, we could do a nice visualization of what uncertainty sampling actually does. 

Instead, let us do the following:
1. Graph the decision distribution each time.
2. See if it learns something reasonable.
3. Instead, let the RL paint a picture of itself, and this will evolve over time.

Therefore, we will graph the decision distribution, and see the fixed convergence point. 

The major drawbacks I see to this work are the following:
1. weak correlation between action and outcome
2. WAY too much randomness!
3. 

metric learning/nlp for text is better!

so, let it die. 

scaling up, may solve some issues:
1. kmeans may work better in higher dimensions 
2. bigger batch size means it observes bigger impacts of its actions
3. 

Therefore, we propose the following:
1. batch size with 10 samples. That is, we will still have 10 steps. But each of these 10 steps involves sampling 10 points! 

If we force all of them to be the same then... => the gradient updates can be the same (it is just like an amplified signal). One drawback to this approach is the following: we will suffer from the same batch correlation problem. However, the goal is to show faster and better learning when we have more impact. 

In particular, it is expected that we will learn to not policy collapse, even without any explicit penalties or regularization. 

Later on, we can address the batch problem


If we let them differ, then... => (we need to think about how the loss will change)

2. nice visualization

# May 3
Our goal is to simply show that when we can pick a batch of examples, then the uncertainty estimator is not as good. (random will be mostly unaffected)

# May 2

Reframe and re-pitch towards BATCH
right now uncertainty is the home court, since it gets currently uncertain (up to the datapoint!)
That is, we should ask the model for 10 datapoints! 

The most logical extension is to make the output layer be 10* the # of output classes, then reshape. 

First though,we NEED the visualization of probabilities over time.

We have unbounded compute on the BACK end!

In particular what is the influence of a single datapoint? Instead, we should establish graphs showing the 
results of adding more datapoints, and the effect on training loss. 

We need to better motivate the problem! 

Also, this means we need to add like 10 datapoints at once.
We can examine: # training data points vs loss, and # training data points vs accuracy.
From a random sample, we expect to see: ...

If even the random signal is too noisy, then it would not be great. But as our ORACLE shows, we CAN select the best datapoints to get the BEST acc! 

What we want actually is more fundamental work into the decision boundary of a neural network; for instance, can another neural
network predict this decision boundary?

And, again with the influence functions as well. how does selecting a SINGLE point affect the predictions. , the training loss etc/


# May 1
The game needs to be better played. The actions it selects, need to be DIRECTLY correlated with the final effect it has.
If all we can do is ask for a label, then this is not good enough. 
Since, even if we evenly distribute the labels, it won't necessarily work. 
That is, we should try the "perfect" first. 

Instead, we should rethink the game it plays. For instance, let it just decide whether to label or not.
This will follow work by other people!5fimstu            
            
            
            
            # one other thing: doing a value network, or framing it as a regression problem

# April 29
Working on getting the environment assumptions oF RL working! In particular: we need to RESET the environment to some original baseline EACH TIME! 

Resetting:
1. Unlabel all the datapoints, OR JUST DUMP THE TRAIN DATALOADER (OK!)
2. Won't work exactly: since we DO implicitly have some state tracking over time...

Even just defining the problem for this question is challenging! Amazing!


Need to do LEARNABLE actions from FIRST principles: batching, environment reset, etc.
(Lmao curriculum learning myself; joint human curriculum learning in RL curriculum learning!)

Also:
we mainly figure out how important the RL reward function is. (ensuring 0 centered in particular).
Also:
## RELEARNING ON SAMPLED DATA
**THE BIG BET**: we will need some sort of experience replay. SOmething like a "standard" environment of 50 candidates to evaluate, and train on, and prep, and then hoping it will generalize as we see more data!

Stagewise RL; as we get more samples, we get exponentially more sequences. Lots of nice FUNDAMENTAL ways to motivate this problem!

# April 28
Got the policy gradient (correct implementation working).


    # one other thing: doing a value network, or framing it as a regression problem

            # RL seems like the wrong formulation to do AL.
            
            
                # try and predict directly in the data space?
    # try and graph where in the dataset it does it as well.
    # in this case, we would again need some fix of the policy gradient.
    # we can no longer just do an easy cross entropy
    # instead, we would be operating more in the regime of .
    # this is a nice analysis on problems of this type!

    # just about rotating and sculpting to your particular area you want
    # ultra few shot learning with fixed dimension, horizon!
    # contributions: to the field of meta/few shot learning using an active learning with reinforcement learning approach
    # keep on layering intersections, until you get the particular area you want.
    # even the approach of doing few shot learning, using active learning is pretty novel IMO




    # we would be trying to essentially do q learning on the choice of datapoint. but make sure you pick in the data space (not action, but continuous choice of the datapoint)
    # the key is really then, trying to do X
    # we could literally do an entire course of lin alg during the break!

    # really, digging into the problems of policy gradient

    # now let's graph the unlabelled dataset
