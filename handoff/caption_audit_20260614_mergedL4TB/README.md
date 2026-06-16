# Caption Anomaly Audit — merged L4-safe + truebones

Dataset: `data/animo4d_anytop_clean_L4_safe_plus_truebones`

Captions JSON: `data/animo4d_anytop_clean_L4_safe_plus_truebones/motion_texts_by_file_with_codex_drafts.json`

## Outputs

- Full caption-level CSV: `handoff/caption_audit_20260614_mergedL4TB/caption_anomaly_candidates.csv`
- Unique motion-level review list: `handoff/caption_audit_20260614_mergedL4TB/motion_level_review_list.csv`

## Counts

- suspicious caption rows: `7586`
- suspicious unique motions: `3566`
- priority counts: `{'P1': 5538, 'P2': 2038, 'P0': 10}`

### Issue counts

- `P1_OVER_NARRATIVE`: 3279
- `P1_AI_DRAFT_REMAINING`: 1073
- `P2_ANIMAL_LEVEL_GENERIC`: 1049
- `P2_GENERIC_CREATURE_CLASS`: 988
- `P1_SOURCE_ACTION_ARTIFACT`: 961
- `P1_BAD_ARTICLE_A_ANIMAL`: 175
- `P1_REST_POSE`: 50
- `P0_TOO_LONG`: 5
- `P0_TOO_SHORT_PLACEHOLDER`: 4
- `P0_BAD_REPEATED_TOKEN`: 1
- `P2_PUNCT_OR_CASE_ODD`: 1

### Issue by source dataset

- `P0_BAD_REPEATED_TOKEN`: animo4d_L4_safe: 1
- `P0_TOO_LONG`: animo4d_L4_safe: 5
- `P0_TOO_SHORT_PLACEHOLDER`: animo4d_L4_safe: 4
- `P1_AI_DRAFT_REMAINING`: truebones: 1073
- `P1_BAD_ARTICLE_A_ANIMAL`: truebones: 175
- `P1_OVER_NARRATIVE`: animo4d_L4_safe: 3018, truebones: 261
- `P1_REST_POSE`: truebones: 50
- `P1_SOURCE_ACTION_ARTIFACT`: animo4d_L4_safe: 949, truebones: 12
- `P2_ANIMAL_LEVEL_GENERIC`: truebones: 1049
- `P2_GENERIC_CREATURE_CLASS`: animo4d_L4_safe: 30, truebones: 958
- `P2_PUNCT_OR_CASE_ODD`: animo4d_L4_safe: 1

## Highest-priority examples

### P0_BAD_REPEATED_TOKEN

- `train` `animo4d_L4_safe` `PZ_Wisent_Male_wisent_male__animationnotmotionextractedlocomotion_manisetd142891d__wisent_male_matingcourtship_457.npy` cap2 (4w/12296c): 'The male wise NTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNT'

### P0_TOO_LONG

- `train` `animo4d_L4_safe` `PZ_Wisent_Male_wisent_male__animationnotmotionextractedlocomotion_manisetd142891d__wisent_male_matingcourtship_457.npy` cap2 (4w/12296c): 'The male wise NTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNT'
- `train` `animo4d_L4_safe` `PZ_Giant_Otter_Juvenile_giant_otter_juvenile__animationnotmotionextractedlocomotion_manisetd4154e76__giant_otter_juvenile_gimmick_103.npy` cap2 (42w/270c): 'The juvenile giant otter uses its nimble body to dart into the waters edge creating a mesmerizing display of agility as it skillfully navigates around the rocks demonstrating its remarkable ability to move swiftly through the aquatic environment with precision and grace'
- `train` `animo4d_L4_safe` `PZ_Saiga_Male_saiga_male__animationnotmotionextractedfighting_manisetcf7c23b4__saiga_male_fighttauntreact_120.npy` cap2 (41w/241c): 'The male saiga charges forward its muscles tense with anticipation and suddenly it shifts direction to avoid an incoming predators attack causing the predator to pause in confusion and fear as if reacting to a sudden taunt from a wild animal'
- `train` `animo4d_L4_safe` `PZ_Bactrian_Camel_Juvenile_bactrian_camel_juvenile__animationnotmotionextractedlocomotion_manisetecfccfaf__bactrian_camel_juvenile_fightreacttodie_220.npy` cap2 (41w/216c): 'The juvenile bactrian camel fights with another camel but due to its small size it cannot overpower the older one It reacts by trying to escape but it is too late as the other camel attacks again leading to its death'
- `train` `animo4d_L4_safe` `PZ_European_Fallow_Deer_Juvenile_european_fallow_deer_juvenile__animationnotmotionextractedfighting_manisetaa1d0ef7__european_fallow_deer_juvenile_fightvictorytaunt_144.npy` cap2 (31w/189c): 'The juvenile fallow deer fights with great vigor but ultimately succumbs to the predators superior strength and skill leaving behind an eerie taunt of triumph as it retreats into the forest'

### P0_TOO_SHORT_PLACEHOLDER

- `train` `animo4d_L4_safe` `PZ_Tamworth_Pig_Male_tamworth_pig_male__animationnotmotionextractedfighting_manisetdfc6f239__tamworth_pig_male_swimbasetoswimtreadwater_336.npy` cap3 (1w/1c): 'A'
- `train` `animo4d_L4_safe` `PZ_Tamworth_Pig_Male_tamworth_pig_male__animationnotmotionextractedfighting_manisetdfc6f239__tamworth_pig_male_swimbasetoswimtreadwater_336.npy` cap4 (1w/1c): 'B'
- `train` `animo4d_L4_safe` `PZ_Tamworth_Pig_Male_tamworth_pig_male__animationnotmotionextractedfighting_manisetdfc6f239__tamworth_pig_male_swimbasetoswimtreadwater_336.npy` cap5 (1w/1c): 'C'
- `train` `animo4d_L4_safe` `PZ_Tamworth_Pig_Male_tamworth_pig_male__animationnotmotionextractedfighting_manisetdfc6f239__tamworth_pig_male_swimbasetoswimtreadwater_336.npy` cap6 (1w/1c): 'D'

### P1_AI_DRAFT_REMAINING

- `val` `truebones` `Monkey___Attack4_570.npy` cap2 (22w/127c): 'A monkey starts from a ready stance, drives the attacking body part forward or sideways, and then returns toward a stable pose.'
- `val` `truebones` `Trex___hit_leg_976.npy` cap2 (23w/127c): 'A t. rex starts from a ready stance, drives the attacking body part forward or sideways, and then returns toward a stable pose.'
- `val` `truebones` `Trex___hit_torso_right_971.npy` cap2 (23w/127c): 'A t. rex starts from a ready stance, drives the attacking body part forward or sideways, and then returns toward a stable pose.'
- `val` `truebones` `Deer___Gallop_271.npy` cap2 (19w/114c): 'A deer begins in a balanced stance, cycles the legs through the movement, and finishes in another locomotion pose.'
- `val` `truebones` `Elephant___Idle1_316.npy` cap2 (19w/114c): 'An elephant starts in a stable pose, makes small repeated posture adjustments, and ends in nearly the same stance.'
- `val` `truebones` `Fox_-_Run_362.npy` cap2 (19w/113c): 'A fox begins in a balanced stance, cycles the legs through the movement, and finishes in another locomotion pose.'
- `val` `truebones` `Rat___Scamper_750.npy` cap2 (19w/113c): 'A rat begins in a balanced stance, cycles the legs through the movement, and finishes in another locomotion pose.'
- `val` `truebones` `Scorpion-2___stand_ready_851.npy` cap2 (19w/113c): 'A scorpion starts in a stable pose, makes small repeated posture adjustments, and ends in nearly the same stance.'
- `val` `truebones` `Trex___idle_attack_to_run_left_967.npy` cap2 (20w/111c): 'A t. rex starts in a stable pose, makes small repeated posture adjustments, and ends in nearly the same stance.'
- `val` `truebones` `Trex___long_death_992.npy` cap2 (19w/111c): 'A t. rex begins upright or partially supported, loses height through the body, and ends low against the ground.'
- `val` `truebones` `Lynx___Idle4_552.npy` cap2 (19w/109c): 'A lynx starts in a stable pose, makes small repeated posture adjustments, and ends in nearly the same stance.'
- `val` `truebones` `Monkey___Attack4_570.npy` cap1 (18w/98c): 'A monkey shifts weight through the body and uses the head, forelimbs, or tail area for the attack.'
- `val` `truebones` `Trex___hit_leg_976.npy` cap1 (19w/98c): 'A t. rex shifts weight through the body and uses the head, forelimbs, or tail area for the attack.'
- `val` `truebones` `Trex___hit_torso_right_971.npy` cap1 (19w/98c): 'A t. rex shifts weight through the body and uses the head, forelimbs, or tail area for the attack.'
- `val` `truebones` `Elephant___Idle1_316.npy` cap1 (14w/89c): 'An elephant holds a mostly stationary stance while making subtle limb and body movements.'
- `val` `truebones` `Scorpion-2___stand_ready_851.npy` cap1 (14w/88c): 'A scorpion holds a mostly stationary stance while making subtle limb and body movements.'
- `val` `truebones` `Trex___idle_attack_to_run_left_967.npy` cap1 (15w/86c): 'A t. rex holds a mostly stationary stance while making subtle limb and body movements.'
- `val` `truebones` `Lynx___Idle4_552.npy` cap1 (14w/84c): 'A lynx holds a mostly stationary stance while making subtle limb and body movements.'
- `val` `truebones` `Trex___long_death_992.npy` cap1 (17w/83c): 'A t. rex lowers the torso and limbs toward the ground in a falling or dying motion.'
- `val` `truebones` `Deer___Gallop_271.npy` cap1 (13w/74c): 'A deer alternates its limbs while the torso shifts through a forward gait.'

### P1_BAD_ARTICLE_A_ANIMAL

- `val` `truebones` `Lynx___Idle4_552.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'
- `val` `truebones` `Scorpion-2___stand_ready_851.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'
- `val` `truebones` `Trex___idle_attack_to_run_left_967.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'
- `val` `truebones` `Monkey___Attack4_570.npy` cap3 (9w/47c): 'A animal lunges or strikes in an attack motion.'
- `val` `truebones` `Deer___Gallop_271.npy` cap3 (8w/44c): 'A animal gallops through a short gait cycle.'
- `val` `truebones` `Trex___long_death_992.npy` cap3 (9w/44c): 'A animal collapses or falls into a low pose.'
- `val` `truebones` `Fox_-_Run_362.npy` cap3 (8w/41c): 'A animal runs through a short gait cycle.'
- `val` `truebones` `Rat___Scamper_750.npy` cap3 (8w/41c): 'A animal runs through a short gait cycle.'
- `val` `truebones` `Trex___hit_leg_976.npy` cap3 (7w/39c): 'A animal strikes forward with its body.'
- `val` `truebones` `Trex___hit_torso_right_971.npy` cap3 (7w/39c): 'A animal strikes forward with its body.'
- `train` `truebones` `Bird___FlyFast_116.npy` cap3 (11w/63c): 'A animal moves through a flying motion with its wings extended.'
- `train` `truebones` `Pigeon___FlyLoop_612.npy` cap3 (11w/63c): 'A animal moves through a flying motion with its wings extended.'
- `train` `truebones` `Spider___Landing_925.npy` cap3 (11w/63c): 'A animal moves through a flying motion with its wings extended.'
- `train` `truebones` `Tukan___FlyLoop_1043.npy` cap3 (11w/63c): 'A animal moves through a flying motion with its wings extended.'
- `train` `truebones` `Centipede___Idle3_203.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'
- `train` `truebones` `Cricket___IdlePissed_249.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'
- `train` `truebones` `Cricket___Idle_242.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'
- `train` `truebones` `Fox_-_Idle1_360.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'
- `train` `truebones` `Fox_-_Idle2_356.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'
- `train` `truebones` `Fox_-_Idle3_368.npy` cap3 (10w/56c): 'A animal holds an idle pose with small body adjustments.'

### P1_OVER_NARRATIVE

- `val` `animo4d_L4_safe` `PZ_Saiga_Male_saiga_male__animationmotionextractedlocomotion_maniset23741078__saiga_male_fightreacttodie_8.npy` cap2 (23w/121c): 'The male saiga charges its opponent in a fight but it is too late as it dies from injuries sustained during the encounter'
- `val` `truebones` `Raptor3___AttackRight_737.npy` cap3 (20w/121c): 'The prehistoric predator twists its body to the right and lunges forward with its right foot, simulating a biting attack.'
- `val` `animo4d_L4_safe` `PZ_California_Sea_Lion_Male_california_sea_lion_male__animationnotmotionextractedfighting_maniset21312b4e__california_sea_lion_male_fighttauntreact_150.npy` cap2 (16w/100c): 'The male sea lion fights with aggression while taunting his opponent causing them to react nervously'
- `val` `truebones` `BrownBear___Attack_122.npy` cap1 (18w/98c): 'An apex predator stands on all fours, rears up, and lunges forward to bite with its powerful jaws.'
- `val` `animo4d_L4_safe` `PZ_Nine_Banded_Armadillo_Juvenile_nine_banded_armadillo_juvenile__animationnotmotionextractedlocomotion_maniset604a44af__nine_banded_armadillo_juvenile_fightreacttodie_188.npy` cap2 (15w/96c): 'The juvenile banded armadillo fights with another armadillo reacting to death by fleeing in fear'
- `val` `truebones` `Raptor2___EggTend_704.npy` cap3 (15w/95c): 'A prehistoric predator quickly picks up food from the ground and remains vigilant while eating.'
- `val` `truebones` `Raptor___FastWalk_689.npy` cap3 (16w/95c): 'A prehistoric predator is walking quickly with its head held high and tail swaying for balance.'
- `val` `truebones` `Tyranno___Attack_1064.npy` cap4 (14w/95c): 'A prehistoric predator prepares to attack by raising its head and striking downward powerfully.'
- `val` `animo4d_L4_safe` `PZ_Reticulated_Giraffe_Male_reticulated_giraffe_male__animationnotmotionextractedlocomotion_maniset2a56d1e7__reticulated_giraffe_male_fighttauntreact_123.npy` cap1 (14w/93c): 'The male reticulated giraffe fights then taunts his opponent before reacting to the situation'
- `val` `truebones` `Raptor2___IdleAlert_698.npy` cap3 (14w/93c): 'A prehistoric predator stands still, breathing with subtle body movements and a swaying tail.'
- `val` `animo4d_L4_safe` `PZ_Nine_Banded_Armadillo_Juvenile_nine_banded_armadillo_juvenile__animationnotmotionextractedlocomotion_maniset604a44af__nine_banded_armadillo_juvenile_fightreacttodie_188.npy` cap1 (14w/92c): 'The juvenile banded armadillo fights and then reacts to its opponent before ultimately dying'
- `val` `animo4d_L4_safe` `PZ_Bengal_Tiger_Female_bengal_tiger_female__animationnotmotionextractedfighting_maniset1a246b35__bengal_tiger_female_fighttauntreact_345.npy` cap1 (15w/91c): 'The female Bengal tiger fights then taunts her opponent and finally reacts to the situation'
- `val` `animo4d_L4_safe` `PZ_Dromedary_Camel_Female_dromedary_camel_female__animationmotionextractedlocomotion_maniseta3492670__dromedary_camel_female_fighttauntreact_7.npy` cap1 (14w/91c): 'The female dromedary camel fights then taunts her opponent before reacting to the situation'
- `val` `truebones` `Crocodile___Idle_257.npy` cap0 (15w/90c): 'An ambush predator slightly opens its mouth and gently wags its tail while facing forward.'
- `val` `animo4d_L4_safe` `PZ_North_American_Beaver_Juvenile_north_american_beaver_juvenile__animationnotmotionextractedfighting_maniset89bc070d__north_american_beaver_juvenile_gimmick_158.npy` cap1 (13w/89c): 'The juvenile American beaver swims across the pond before performing a remarkable gimmick'
- `val` `animo4d_L4_safe` `PZ_Pygmy_Hippo_Female_pygmy_hippo_female__animationnotmotionextractedlocomotion_maniset482fb3e4__pygmy_hippo_female_fightvictorytaunt_205.npy` cap0 (15w/89c): 'The female pygmy hippo engages in a fight emerges victorious and then taunts her opponent'
- `val` `animo4d_L4_safe` `PZ_Saiga_Juvenile_saiga_juvenile__animationnotmotionextractedfighting_maniset28b0fb57__saiga_juvenile_fighttaunt_99.npy` cap2 (14w/88c): 'The juvenile saiga charges forward and then retreats in fear while taunting its predator'
- `val` `animo4d_L4_safe` `PZ_Caracal_Juvenile_caracal_juvenile__animationmotionextractedfighting_pounce_maniset2d5e358__caracal_juvenile_fighttaunttostand_14.npy` cap2 (14w/87c): 'The juvenile caracal fights with its teeth and stands still while taunting its opponent'
- `val` `animo4d_L4_safe` `PZ_Southern_White_Rhinoceros_Female_southern_white_rhinoceros_female__animationnotmotionextractedfighting_maniset762bed32__southern_white_rhinoceros_female_fightreacttodie_102.npy` cap1 (13w/86c): 'The female white rhinoceros fights before reacting to her opponent and ultimately dies'
- `val` `animo4d_L4_safe` `PZ_Wolverine_Male_wolverine_male__animationnotmotionextractedlocomotion_manisetedaaad1f__wolverine_male_fighttauntreact_686.npy` cap1 (14w/86c): 'The male wolverine fights then taunts his opponent and finally reacts to the situation'

### P1_REST_POSE

- `val` `truebones` `Scorpion-2___Idle_850.npy` cap2 (11w/66c): 'Rest pose of a desert invertebrate with its tail drooping forward.'
- `val` `truebones` `Scorpion-2___Idle_850.npy` cap3 (10w/60c): 'Rest pose of an eight-legged with its tail drooping forward.'
- `val` `truebones` `Scorpion-2___Idle_850.npy` cap4 (10w/59c): 'Rest pose of an exoskeletal with its tail drooping forward.'
- `val` `truebones` `Scorpion-2___Idle_850.npy` cap1 (10w/56c): 'Rest pose of an arachnid with its tail drooping forward.'
- `val` `truebones` `Scorpion-2___Idle_850.npy` cap0 (10w/54c): 'Rest pose of an animal with its tail drooping forward.'
- `train` `truebones` `Anaconda___Tired_34.npy` cap2 (11w/60c): 'Rest pose of an aquatic serpent with its head drooping down.'
- `train` `truebones` `Raindeer___Idle_670.npy` cap3 (10w/56c): 'Rest pose of a hoofed mammal with widely spread antlers.'
- `train` `truebones` `Anaconda___Tired_34.npy` cap4 (10w/55c): 'Rest pose of a constrictor with its head drooping down.'
- `train` `truebones` `Anaconda___Tired_34.npy` cap0 (10w/53c): 'Rest pose of an anaconda with its head drooping down.'
- `train` `truebones` `Raindeer___Idle_670.npy` cap2 (9w/52c): 'Rest pose of a herbivore with widely spread antlers.'
- `train` `truebones` `Raindeer___Idle_670.npy` cap4 (9w/52c): 'Rest pose of a quadruped with widely spread antlers.'
- `train` `truebones` `Anaconda___Tired_34.npy` cap1 (10w/51c): 'Rest pose of an animal with its head drooping down.'
- `train` `truebones` `Raindeer___Idle_670.npy` cap0 (9w/50c): 'Rest pose of an animal with widely spread antlers.'
- `train` `truebones` `Raindeer___Idle_670.npy` cap1 (9w/50c): 'Rest pose of a caribou with widely spread antlers.'
- `train` `truebones` `Anaconda___Tired_34.npy` cap3 (10w/48c): 'Rest pose of a boid with its head drooping down.'
- `train` `truebones` `Ostrich___Idle2_581.npy` cap4 (6w/37c): 'Rest pose of a ground-dwelling avian.'
- `train` `truebones` `Camel___IdleLoop_180.npy` cap3 (6w/35c): 'Rest pose of an even-toed ungulate.'
- `train` `truebones` `Camel___IdleLoop_180.npy` cap2 (6w/32c): 'Rest pose of a desert quadruped.'
- `train` `truebones` `Crocodile___SleepLoop_258.npy` cap0 (6w/32c): 'Rest pose of an ambush predator.'
- `train` `truebones` `Bird___IdleLoop_107.npy` cap2 (6w/31c): 'Rest pose of a beaked creature.'

### P1_SOURCE_ACTION_ARTIFACT

- `val` `animo4d_L4_safe` `PZ_Meerkat_Juvenile_meerkat_juvenile__animationnotmotionextractedlocomotion_maniset9f3e78c0__meerkat_juvenile_gimmickloop02_161.npy` cap0 (17w/110c): 'The juvenile meerkat performs a playful gimmick by scampering around and then pausing to look up inquisitively'
- `val` `animo4d_L4_safe` `PZ_Asian_Small_Clawed_Otter_Juvenile_asian_small_clawed_otter_juvenile__animationnotmotionextractedlocomotion_maniset3547edfb__asian_small_clawed_otter_juvenile_gimmick_252.npy` cap0 (19w/107c): 'The juvenile clawed otter performs a playful gimmick by diving into the water and then emerging with a fish'
- `val` `animo4d_L4_safe` `PZ_Tasmanian_Devil_Female_tasmanian_devil_female__animationnotmotionextractedbehaviour_maniset2c94a132__tasmanian_devil_female_standgimmick_163.npy` cap1 (15w/98c): 'The female Tasmanian devil stands still pretending to be a gimmick to distract potential predators'
- `val` `animo4d_L4_safe` `PZ_Bactrian_Camel_Male_bactrian_camel_male__animationnotmotionextractedlocomotion_manisetef04758c__bactrian_camel_male_standgimmick01_252.npy` cap1 (16w/90c): 'The male bactrian camel stands still but then suddenly freezes in place due to the gimmick'
- `val` `animo4d_L4_safe` `PZ_North_American_Beaver_Juvenile_north_american_beaver_juvenile__animationnotmotionextractedfighting_maniset89bc070d__north_american_beaver_juvenile_gimmick_158.npy` cap1 (13w/89c): 'The juvenile American beaver swims across the pond before performing a remarkable gimmick'
- `val` `animo4d_L4_safe` `PZ_Capuchin_Monkey_Male_capuchin_monkey_male__animationnotmotionextractedbehaviour_maniset273ded7e__capuchin_monkey_male_gimmick_168.npy` cap0 (13w/85c): 'The male capuchin monkey performs a playful gimmick by leaping and then somersaulting'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Male_meerkat_male__animationmotionextractedbehaviour_maniset5f721adf__meerkat_male_burrowgimmick_1.npy` cap0 (15w/80c): 'The male meerkat performs a burrow gimmick by darting in and out of the entrance'
- `val` `animo4d_L4_safe` `PZ_Standard_Donkey_Male_standard_donkey_male__animationmotionextractedlocomotion_maniset6569664c__standard_donkey_male_gimmick_99.npy` cap0 (12w/74c): 'The male standard donkey performs a playful gimmick by trotting in circles'
- `val` `animo4d_L4_safe` `PZ_Scimitar_Horned_Oryx_Male_scimitar_horned_oryx_male__animationnotmotionextractedbehaviour_maniset617be95b__scimitar_horned_oryx_male_standgimmick01_59.npy` cap1 (12w/72c): 'The male horned oryx stands still using his gimmick to remain motionless'
- `val` `animo4d_L4_safe` `PZ_Black_Wildebeest_Female_black_wildebeest_female__animationnotmotionextractedbehaviour_maniset31aeadb9__black_wildebeest_female_standgimmick01_85.npy` cap1 (11w/67c): 'The female black wildebeest stands still pretending to be a gimmick'
- `val` `animo4d_L4_safe` `PZ_Somali_Wild_Ass_Male_somali_wild_ass_male__animationnotmotionextractedfighting_maniset6fedd2c5__somali_wild_ass_male_gimmick_291.npy` cap0 (12w/67c): 'The male wild ass performs a playful gimmick by trotting in circles'
- `val` `animo4d_L4_safe` `PZ_Dingo_Female_dingo_female__animationmotionextractedfighting_manisetc6604663__dingo_female_standgimmick_5.npy` cap1 (11w/66c): 'The female dingo stands still using a gimmick to remain motionless'
- `val` `animo4d_L4_safe` `PZ_Prairie_Dog_Male_prairie_dog_male__animationnotmotionextractedbehaviour_maniseta0ee1e22__prairie_dog_male_standtogimmickloop_85.npy` cap1 (11w/65c): 'The male prairie dog stands still before scurrying to the gimmick'
- `val` `animo4d_L4_safe` `PZ_Bactrian_Camel_Male_bactrian_camel_male__animationnotmotionextractedlocomotion_manisetef04758c__bactrian_camel_male_standgimmick01_252.npy` cap0 (10w/63c): 'The male bactrian camel stands still employing a clever gimmick'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Juvenile_meerkat_juvenile__animationnotmotionextractedlocomotion_maniset9f3e78c0__meerkat_juvenile_gimmickloop02_161.npy` cap1 (12w/63c): 'The juvenile meerkat uses a gimmick to slide down a sandy slope'
- `val` `animo4d_L4_safe` `PZ_Asian_Small_Clawed_Otter_Juvenile_asian_small_clawed_otter_juvenile__animationnotmotionextractedlocomotion_maniset3547edfb__asian_small_clawed_otter_juvenile_gimmick_252.npy` cap1 (11w/62c): 'The juvenile clawed otter slips and then slides down a gimmick'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Male_meerkat_male__animationnotmotionextractedlocomotion_maniset8d623e1d__meerkat_male_standtogimmickloop_173.npy` cap1 (10w/62c): 'The male meerkat stands still before scampering to the gimmick'
- `val` `animo4d_L4_safe` `PZ_Babirusa_Male_babirusa_male__animationnotmotionextractedfighting_manisetc0f1e1ee__babirusa_male_standgimmick01_148.npy` cap0 (11w/61c): 'The male babirusa stands still as part of its playful gimmick'
- `val` `animo4d_L4_safe` `PZ_Blue_Wildebeest_Juvenile_blue_wildebeest_juvenile__animationnotmotionextractedbehaviour_manisete1d7cafa__blue_wildebeest_juvenile_standgimmick01_87.npy` cap2 (10w/61c): 'The juvenile blue wildebeest stands still in its gimmick pose'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Male_meerkat_male__animationmotionextractedbehaviour_maniset5f721adf__meerkat_male_burrowgimmick_1.npy` cap2 (10w/61c): 'The male meerkat burrows and then performs his burrow gimmick'

### P2_ANIMAL_LEVEL_GENERIC

- `val` `truebones` `Isopetra___Idle2_475.npy` cap0 (23w/127c): 'An animal stands still on the ground, occasionally twitching its legs and moving its body and tail up and down as if breathing.'
- `val` `truebones` `Raptor3___AttackRight_737.npy` cap0 (19w/107c): 'The animal twists its body to the right and lunges forward with its right foot, simulating a biting attack.'
- `val` `truebones` `SabreToothTiger___Sitting2_804.npy` cap0 (19w/104c): 'The animal sits on the ground with its hind legs extended forward and head slightly turned to the right.'
- `val` `truebones` `Gazelle___HeadPoke_382.npy` cap0 (18w/99c): 'An animal performs a head poke motion by bending its knees and thrusting its head and horns upward.'
- `val` `truebones` `BrownBear___Attack_122.npy` cap0 (17w/91c): 'An animal stands on all fours, rears up, and lunges forward to bite with its powerful jaws.'
- `val` `truebones` `Giantbee___Attack2_387.npy` cap0 (15w/91c): 'The animal performs an aggressive attack with rapid wing flaps and a powerful sting motion.'
- `val` `truebones` `Leapord___Attack_512.npy` cap0 (15w/90c): 'An animal attacks by raising its upper body and front paws, then striking down forcefully.'
- `val` `truebones` `SandMouse___Die3_822.npy` cap0 (13w/90c): 'An animal is struck head-on and flung backward, eventually losing consciousness and dying.'
- `val` `truebones` `Centipede___Attack_202.npy` cap0 (16w/89c): 'An animal prepares and executes an attack by raising its body and striking with its arms.'
- `val` `truebones` `Spider___barking_918.npy` cap0 (16w/89c): 'An animal raises its body and front legs, spreading its jaws to make a threatening sound.'
- `val` `truebones` `Cricket___Attack2_245.npy` cap0 (15w/88c): 'An animal steps back, raises its body and pincers, and strikes forward with its pincers.'
- `val` `truebones` `Roach___Shake_769.npy` cap0 (15w/88c): 'An animal shakes its body, spreading its wings and moving its head and antennae rapidly.'
- `val` `truebones` `SabreToothTiger___GetUp_810.npy` cap0 (15w/88c): 'An animal rises from a prone position by extending its hind and front legs sequentially.'
- `val` `truebones` `Comodoa___Idle_213.npy` cap0 (16w/87c): 'An animal stands on all fours, moving its tail side to side and shaking its midsection.'
- `val` `truebones` `Crab___Attack2_237.npy` cap0 (13w/84c): 'An animal hesitates briefly before assuming an attacking stance with its right claw.'
- `val` `truebones` `Flamingo_Flamingo_OneLEgBEnt_355.npy` cap0 (16w/84c): 'An animal stands on its left leg while flapping its wings and bending its right leg.'
- `val` `truebones` `Raptor2___EggTend_704.npy` cap0 (14w/82c): 'An animal quickly picks up food from the ground and remains vigilant while eating.'
- `val` `truebones` `Raptor___FastWalk_689.npy` cap0 (15w/82c): 'An animal is walking quickly with its head held high and tail swaying for balance.'
- `val` `truebones` `Tyranno___Attack_1064.npy` cap0 (13w/82c): 'An animal prepares to attack by raising its head and striking downward powerfully.'
- `val` `truebones` `Ant___Sting2_46.npy` cap0 (15w/81c): 'An animal prepares to sting by pulling its body down and raising its tail upward.'

### P2_GENERIC_CREATURE_CLASS

- `val` `truebones` `Isopetra___Idle2_475.npy` cap4 (24w/133c): 'A social insect stands still on the ground, occasionally twitching its legs and moving its body and tail up and down as if breathing.'
- `val` `truebones` `Isopetra___Idle2_475.npy` cap2 (23w/127c): 'An insect stands still on the ground, occasionally twitching its legs and moving its body and tail up and down as if breathing.'
- `val` `truebones` `SabreToothTiger___Sitting2_804.npy` cap1 (20w/116c): 'The carnivorous mammal sits on the ground with its hind legs extended forward and head slightly turned to the right.'
- `val` `truebones` `Gazelle___HeadPoke_382.npy` cap3 (19w/105c): 'A hoofed mammal performs a head poke motion by bending its knees and thrusting its head and horns upward.'
- `val` `truebones` `BrownBear___Attack_122.npy` cap4 (18w/102c): 'An omnivorous mammal stands on all fours, rears up, and lunges forward to bite with its powerful jaws.'
- `val` `truebones` `Comodoa___Idle_213.npy` cap2 (17w/101c): 'A carnivorous quadruped stands on all fours, moving its tail side to side and shaking its midsection.'
- `val` `truebones` `Gazelle___HeadPoke_382.npy` cap4 (18w/101c): 'A quadruped performs a head poke motion by bending its knees and thrusting its head and horns upward.'
- `val` `truebones` `Leapord___Attack_512.npy` cap2 (16w/101c): 'A carnivorous mammal attacks by raising its upper body and front paws, then striking down forcefully.'
- `val` `truebones` `Centipede___Attack_202.npy` cap3 (17w/99c): 'A crawling creature prepares and executes an attack by raising its body and striking with its arms.'
- `val` `truebones` `SabreToothTiger___GetUp_810.npy` cap1 (16w/99c): 'A carnivorous mammal rises from a prone position by extending its hind and front legs sequentially.'
- `val` `truebones` `Comodoa___Idle_213.npy` cap1 (17w/93c): 'An apex reptile stands on all fours, moving its tail side to side and shaking its midsection.'
- `val` `truebones` `SandMouse___Die3_822.npy` cap3 (13w/92c): 'A quadruped is struck head-on and flung backward, eventually losing consciousness and dying.'
- `val` `truebones` `Giantbee___Attack2_387.npy` cap4 (15w/91c): 'The insect performs an aggressive attack with rapid wing flaps and a powerful sting motion.'
- `val` `truebones` `Cat_CAT_IdlePurr_196.npy` cap4 (14w/88c): 'A domesticated mammal lies down and purrs, occasionally moving its head, ears, and tail.'
- `val` `truebones` `Cricket___Attack2_245.npy` cap4 (15w/88c): 'An insect steps back, raises its body and pincers, and strikes forward with its pincers.'
- `val` `truebones` `Roach___Shake_769.npy` cap2 (15w/88c): 'An insect shakes its body, spreading its wings and moving its head and antennae rapidly.'
- `val` `truebones` `Tricera___Idle_1033.npy` cap3 (14w/86c): 'A horned reptile stands still, appearing to rest, with slight head and body movements.'
- `val` `truebones` `Crocodile___Idle_257.npy` cap4 (14w/83c): 'A quadruped slightly opens its mouth and gently wags its tail while facing forward.'
- `val` `truebones` `Ant___Sting2_46.npy` cap3 (15w/81c): 'An insect prepares to sting by pulling its body down and raising its tail upward.'
- `val` `truebones` `Goat___HoofScrape_394.npy` cap4 (15w/81c): 'A hoofed mammal turns its head left, then scrapes the ground with its right hoof.'

### P2_PUNCT_OR_CASE_ODD

- `train` `animo4d_L4_safe` `PZ_Wisent_Male_wisent_male__animationnotmotionextractedlocomotion_manisetd142891d__wisent_male_matingcourtship_457.npy` cap2 (4w/12296c): 'The male wise NTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNT'

