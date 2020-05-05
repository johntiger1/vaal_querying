# May 4
The goal right now is the following.
1. we have shown that *some* learning may occur.
2. we have also shown that batching is unaddressed by uncertainty.

Therefore, we foresee a use case for the RL if we can produce K outputs, and then we reshape. We can do both a sequential, as well as a batch version. 

That is: in one approach, we produce N\*K outputs, and then reshape, and train. In the other one, we simply retrain sequentially, stepping the network each time...

Less principled, but...

We will *specifically* show that our RL network solves the batched problem, and ONLY the batched problem

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
