# May 2
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