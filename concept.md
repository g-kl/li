In this lineage-inpainting directory is the publication version of the repo for lineage inpainting (the development version, with outputs etc. is in pool-and-split). We've submitted the paper and it is now under-review. I'm thinking of doing some "pre-revision" overhaul of this, and making a new streamlined version in this repo (li). My major concerns are the following:

1. The codebase is so crazy and complicated with unnecessary stuff at every step of the way. We should probably go all the way down to the TreeRecord/TreeBatch and pare out and simplify. At the end, we should be able to have a simple notebook where we import things and train illustratively (I think this would be very ugly in the current setup).

2. I don't think we actually have clear ways of interacting with the model. What is the output one should interact with? I suppose it's the couplings in assignments. We need to have a good interface for this, I think. And we should probably take multiple samples.

3. I want to build confidence that our method isn't worse than some trivial baselines. I think our synthetic data experiments are strong and build confidence, but for the real data the QC metrics for the couplings are quite bad. Is there any way to either (1) improve the couplings or (2) have some way of looking at the model outputs to convince ourselves (and others) that they are not silly?

4. The perturbation pipeline is ugly. Is there a way we can simplify things? Currently, to choose targets and make real predictions, we have to do a lot of filtering I feel. We make a lot of heuristic choices like freeze vs. prune, off-manifold fraction. We look at the model predictions then interpret it ourselves: is this a weird gene? is there literature support? etc. Perhaps we can standardize this somehow? Perhaps embed it in a Claude skill?

5. The analysis and figure making code is quite messy. I am making a package (see the polyplot repo) to streamline plotting, but we should also standardize analysis code a bit.

Of course since we submitted the paper, we don't want to dramatically change anything. What I imagine is a sort of overhaul of the codebase, refit all the models. Then I imagine none of our main text figures will actually change that much (maybe small numbers shift, but conceptually the same). The supplement details and ablations may change a little bit, and we may add a few pieces, but the story stays the same.

How would you simplify/revamp/restructure the codebase? How would you make artifacts that other scientists can interact with? How would you build confidence that the predictions aren't horrifyingly bad? How would you simplify/formalize the perturbation prediction?

Our goals here are to:

1. create a roadmap.md file which will lay out our plan -- including more long term stuff for the to-do list
2. create the intial core part of the code (no analysis, but the main data and hydra structure).
3. set up some checks that the tree data structure works as we expect
4. set up some intial metrics (at least germ layer crossing)
5. fit our "main" model and check that we can get a good score for the metric