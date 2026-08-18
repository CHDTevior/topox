Derived from data/holdout_topologies_v1.json
sha256=0baf7bcfb82266d504f9bb45d0ec4f22980043ee49e53c0d7d13b40ebc858e0c

*_retained.json : retained canonical topologies only. train_main_retained is the ONLY
                  evaluator training set; val_all_retained is the ONLY signal allowed to
                  select its checkpoint.
held_*.json     : TEST QUERIES, used only after the evaluator AND the generator are
                  frozen. No checkpoint, threshold, architecture or normalisation may be
                  tuned against them.
