# Hard Caption Corruption Audit

Dataset: `data/animo4d_anytop_clean_L4_safe_plus_truebones`

This narrow audit only flags hard corruption or low-semantic artifact captions. It intentionally ignores grammar quality, narrative style, and normal long-but-meaningful descriptions.

## Files

- Caption-level CSV: `handoff/caption_audit_20260614_mergedL4TB_hard_corrupt/hard_corrupt_caption_candidates.csv`
- Motion-level CSV: `handoff/caption_audit_20260614_mergedL4TB_hard_corrupt/hard_corrupt_motion_review_list.csv`

## Counts

- caption rows: `1291`
- unique motions: `501`

- `SOURCE_OR_ACTION_ARTIFACT`: 961
- `LOW_SEMANTIC_TEMPLATE`: 324
- `PLACEHOLDER_ONE_TOKEN`: 4
- `EXTREME_LENGTH_GT_1000_CHARS`: 1
- `REPEATED_TOKEN_ARTIFACT`: 1

## Examples

### EXTREME_LENGTH_GT_1000_CHARS

- `train` `animo4d_L4_safe` `PZ_Wisent_Male_wisent_male__animationnotmotionextractedlocomotion_manisetd142891d__wisent_male_matingcourtship_457.npy` cap2 (4w/12296c): 'The male wise NTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNT'

### REPEATED_TOKEN_ARTIFACT

- `train` `animo4d_L4_safe` `PZ_Wisent_Male_wisent_male__animationnotmotionextractedlocomotion_manisetd142891d__wisent_male_matingcourtship_457.npy` cap2 (4w/12296c): 'The male wise NTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNTNT'

### PLACEHOLDER_ONE_TOKEN

- `train` `animo4d_L4_safe` `PZ_Tamworth_Pig_Male_tamworth_pig_male__animationnotmotionextractedfighting_manisetdfc6f239__tamworth_pig_male_swimbasetoswimtreadwater_336.npy` cap3 (1w/1c): 'A'
- `train` `animo4d_L4_safe` `PZ_Tamworth_Pig_Male_tamworth_pig_male__animationnotmotionextractedfighting_manisetdfc6f239__tamworth_pig_male_swimbasetoswimtreadwater_336.npy` cap4 (1w/1c): 'B'
- `train` `animo4d_L4_safe` `PZ_Tamworth_Pig_Male_tamworth_pig_male__animationnotmotionextractedfighting_manisetdfc6f239__tamworth_pig_male_swimbasetoswimtreadwater_336.npy` cap5 (1w/1c): 'C'
- `train` `animo4d_L4_safe` `PZ_Tamworth_Pig_Male_tamworth_pig_male__animationnotmotionextractedfighting_manisetdfc6f239__tamworth_pig_male_swimbasetoswimtreadwater_336.npy` cap6 (1w/1c): 'D'

### SOURCE_OR_ACTION_ARTIFACT

- `val` `animo4d_L4_safe` `PZ_Alpine_Goat_Female_alpine_goat_female__animationnotmotionextractedbehaviour_manisetd02b5790__alpine_goat_female_standgimmick01_70.npy` cap0 (9w/48c): 'The female alpine goat stands still as a gimmick'
- `val` `animo4d_L4_safe` `PZ_Arctic_Wolf_Female_arctic_wolf_female__animationnotmotionextractedfighting_maniset2001d71b__arctic_wolf_female_standgimmick01_169.npy` cap0 (9w/48c): 'The female arctic wolf stands still as a gimmick'
- `val` `animo4d_L4_safe` `PZ_Asian_Small_Clawed_Otter_Juvenile_asian_small_clawed_otter_juvenile__animationnotmotionextractedlocomotion_maniset3547edfb__asian_small_clawed_otter_juvenile_gimmick_252.npy` cap0 (19w/107c): 'The juvenile clawed otter performs a playful gimmick by diving into the water and then emerging with a fish'
- `val` `animo4d_L4_safe` `PZ_Asian_Small_Clawed_Otter_Juvenile_asian_small_clawed_otter_juvenile__animationnotmotionextractedlocomotion_maniset3547edfb__asian_small_clawed_otter_juvenile_gimmick_252.npy` cap1 (11w/62c): 'The juvenile clawed otter slips and then slides down a gimmick'
- `val` `animo4d_L4_safe` `PZ_Babirusa_Male_babirusa_male__animationnotmotionextractedfighting_manisetc0f1e1ee__babirusa_male_standgimmick01_148.npy` cap0 (11w/61c): 'The male babirusa stands still as part of its playful gimmick'
- `val` `animo4d_L4_safe` `PZ_Babirusa_Male_babirusa_male__animationnotmotionextractedfighting_manisetc0f1e1ee__babirusa_male_standgimmick01_148.npy` cap1 (9w/59c): 'The male babirusa stands still before employing his gimmick'
- `val` `animo4d_L4_safe` `PZ_Bactrian_Camel_Female_bactrian_camel_female__animationnotmotionextractedfighting_maniset2b182a71__bactrian_camel_female_standgimmick01_169.npy` cap0 (9w/51c): 'The female Bactrian camel stands still as a gimmick'
- `val` `animo4d_L4_safe` `PZ_Bactrian_Camel_Juvenile_bactrian_camel_juvenile__animationnotmotionextractedfighting_maniset6aa2e2c9__bactrian_camel_juvenile_standgimmick01_160.npy` cap0 (9w/53c): 'The juvenile bactrian camel stands still as a gimmick'
- `val` `animo4d_L4_safe` `PZ_Bactrian_Camel_Male_bactrian_camel_male__animationnotmotionextractedlocomotion_manisetef04758c__bactrian_camel_male_standgimmick01_252.npy` cap0 (10w/63c): 'The male bactrian camel stands still employing a clever gimmick'
- `val` `animo4d_L4_safe` `PZ_Bactrian_Camel_Male_bactrian_camel_male__animationnotmotionextractedlocomotion_manisetef04758c__bactrian_camel_male_standgimmick01_252.npy` cap1 (16w/90c): 'The male bactrian camel stands still but then suddenly freezes in place due to the gimmick'
- `val` `animo4d_L4_safe` `PZ_Black_Wildebeest_Female_black_wildebeest_female__animationnotmotionextractedbehaviour_maniset31aeadb9__black_wildebeest_female_standgimmick01_85.npy` cap0 (9w/53c): 'The female black wildebeest stands still as a gimmick'
- `val` `animo4d_L4_safe` `PZ_Black_Wildebeest_Female_black_wildebeest_female__animationnotmotionextractedbehaviour_maniset31aeadb9__black_wildebeest_female_standgimmick01_85.npy` cap1 (11w/67c): 'The female black wildebeest stands still pretending to be a gimmick'
- `val` `animo4d_L4_safe` `PZ_Blue_Wildebeest_Juvenile_blue_wildebeest_juvenile__animationnotmotionextractedbehaviour_manisete1d7cafa__blue_wildebeest_juvenile_standgimmick01_87.npy` cap0 (9w/54c): 'The juvenile blue wildebeest stands still as a gimmick'
- `val` `animo4d_L4_safe` `PZ_Blue_Wildebeest_Juvenile_blue_wildebeest_juvenile__animationnotmotionextractedbehaviour_manisete1d7cafa__blue_wildebeest_juvenile_standgimmick01_87.npy` cap2 (10w/61c): 'The juvenile blue wildebeest stands still in its gimmick pose'
- `val` `animo4d_L4_safe` `PZ_Capuchin_Monkey_Male_capuchin_monkey_male__animationnotmotionextractedbehaviour_maniset273ded7e__capuchin_monkey_male_gimmick_168.npy` cap0 (13w/85c): 'The male capuchin monkey performs a playful gimmick by leaping and then somersaulting'
- `val` `animo4d_L4_safe` `PZ_Capuchin_Monkey_Male_capuchin_monkey_male__animationnotmotionextractedbehaviour_maniset273ded7e__capuchin_monkey_male_gimmick_168.npy` cap1 (7w/43c): 'The male capuchin monkey performs a gimmick'
- `val` `animo4d_L4_safe` `PZ_Dingo_Female_dingo_female__animationmotionextractedfighting_manisetc6604663__dingo_female_standgimmick_5.npy` cap0 (10w/52c): 'The female dingo stands still as part of her gimmick'
- `val` `animo4d_L4_safe` `PZ_Dingo_Female_dingo_female__animationmotionextractedfighting_manisetc6604663__dingo_female_standgimmick_5.npy` cap1 (11w/66c): 'The female dingo stands still using a gimmick to remain motionless'
- `val` `animo4d_L4_safe` `PZ_Indian_Elephant_Female_indian_elephant_female__animationnotmotionextractedfighting_manisetce9752dd__indian_elephant_female_standgimmick01_128.npy` cap0 (9w/52c): 'The female Indian elephant stands still as a gimmick'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Juvenile_meerkat_juvenile__animationnotmotionextractedlocomotion_maniset9f3e78c0__meerkat_juvenile_gimmickloop02_161.npy` cap0 (17w/110c): 'The juvenile meerkat performs a playful gimmick by scampering around and then pausing to look up inquisitively'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Juvenile_meerkat_juvenile__animationnotmotionextractedlocomotion_maniset9f3e78c0__meerkat_juvenile_gimmickloop02_161.npy` cap1 (12w/63c): 'The juvenile meerkat uses a gimmick to slide down a sandy slope'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Male_meerkat_male__animationmotionextractedbehaviour_maniset5f721adf__meerkat_male_burrowgimmick_1.npy` cap0 (15w/80c): 'The male meerkat performs a burrow gimmick by darting in and out of the entrance'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Male_meerkat_male__animationmotionextractedbehaviour_maniset5f721adf__meerkat_male_burrowgimmick_1.npy` cap2 (10w/61c): 'The male meerkat burrows and then performs his burrow gimmick'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Male_meerkat_male__animationnotmotionextractedlocomotion_maniset8d623e1d__meerkat_male_standtogimmickloop_173.npy` cap0 (7w/40c): 'The male meerkat stands still to gimmick'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Male_meerkat_male__animationnotmotionextractedlocomotion_maniset8d623e1d__meerkat_male_standtogimmickloop_173.npy` cap1 (10w/62c): 'The male meerkat stands still before scampering to the gimmick'
- `val` `animo4d_L4_safe` `PZ_North_American_Beaver_Juvenile_north_american_beaver_juvenile__animationnotmotionextractedfighting_maniset89bc070d__north_american_beaver_juvenile_gimmick_158.npy` cap0 (9w/55c): 'The juvenile American beaver swims in a playful gimmick'
- `val` `animo4d_L4_safe` `PZ_North_American_Beaver_Juvenile_north_american_beaver_juvenile__animationnotmotionextractedfighting_maniset89bc070d__north_american_beaver_juvenile_gimmick_158.npy` cap1 (13w/89c): 'The juvenile American beaver swims across the pond before performing a remarkable gimmick'
- `val` `animo4d_L4_safe` `PZ_Prairie_Dog_Male_prairie_dog_male__animationnotmotionextractedbehaviour_maniseta0ee1e22__prairie_dog_male_standtogimmickloop_85.npy` cap0 (8w/44c): 'The male prairie dog stands still to gimmick'
- `val` `animo4d_L4_safe` `PZ_Prairie_Dog_Male_prairie_dog_male__animationnotmotionextractedbehaviour_maniseta0ee1e22__prairie_dog_male_standtogimmickloop_85.npy` cap1 (11w/65c): 'The male prairie dog stands still before scurrying to the gimmick'
- `val` `animo4d_L4_safe` `PZ_Red_Fox_Juvenile_red_fox_juvenile__animationnotmotionextractedfighting_maniset3f9bb538__red_fox_juvenile_gimmick01_68.npy` cap0 (8w/47c): 'The juvenile red fox performs a playful gimmick'

### LOW_SEMANTIC_TEMPLATE

- `val` `animo4d_L4_safe` `PZ_Asian_Small_Clawed_Otter_Juvenile_asian_small_clawed_otter_juvenile__animationnotmotionextractedlocomotion_maniset3547edfb__asian_small_clawed_otter_juvenile_gimmick_252.npy` cap1 (11w/62c): 'The juvenile clawed otter slips and then slides down a gimmick'
- `val` `animo4d_L4_safe` `PZ_Babirusa_Male_babirusa_male__animationnotmotionextractedfighting_manisetc0f1e1ee__babirusa_male_standgimmick01_148.npy` cap0 (11w/61c): 'The male babirusa stands still as part of its playful gimmick'
- `val` `animo4d_L4_safe` `PZ_Babirusa_Male_babirusa_male__animationnotmotionextractedfighting_manisetc0f1e1ee__babirusa_male_standgimmick01_148.npy` cap1 (9w/59c): 'The male babirusa stands still before employing his gimmick'
- `val` `animo4d_L4_safe` `PZ_Bactrian_Camel_Male_bactrian_camel_male__animationnotmotionextractedlocomotion_manisetef04758c__bactrian_camel_male_standgimmick01_252.npy` cap0 (10w/63c): 'The male bactrian camel stands still employing a clever gimmick'
- `val` `animo4d_L4_safe` `PZ_Black_Wildebeest_Female_black_wildebeest_female__animationnotmotionextractedbehaviour_maniset31aeadb9__black_wildebeest_female_standgimmick01_85.npy` cap1 (11w/67c): 'The female black wildebeest stands still pretending to be a gimmick'
- `val` `animo4d_L4_safe` `PZ_Dingo_Female_dingo_female__animationmotionextractedfighting_manisetc6604663__dingo_female_standgimmick_5.npy` cap0 (10w/52c): 'The female dingo stands still as part of her gimmick'
- `val` `animo4d_L4_safe` `PZ_Dingo_Female_dingo_female__animationmotionextractedfighting_manisetc6604663__dingo_female_standgimmick_5.npy` cap1 (11w/66c): 'The female dingo stands still using a gimmick to remain motionless'
- `val` `animo4d_L4_safe` `PZ_Meerkat_Juvenile_meerkat_juvenile__animationnotmotionextractedlocomotion_maniset9f3e78c0__meerkat_juvenile_gimmickloop02_161.npy` cap1 (12w/63c): 'The juvenile meerkat uses a gimmick to slide down a sandy slope'
- `val` `animo4d_L4_safe` `PZ_Prairie_Dog_Male_prairie_dog_male__animationnotmotionextractedbehaviour_maniseta0ee1e22__prairie_dog_male_standtogimmickloop_85.npy` cap1 (11w/65c): 'The male prairie dog stands still before scurrying to the gimmick'
- `val` `animo4d_L4_safe` `PZ_Red_Fox_Juvenile_red_fox_juvenile__animationnotmotionextractedfighting_maniset3f9bb538__red_fox_juvenile_gimmick01_68.npy` cap1 (9w/53c): 'The juvenile red fox performs a gimmick before moving'
- `val` `animo4d_L4_safe` `PZ_Scimitar_Horned_Oryx_Male_scimitar_horned_oryx_male__animationnotmotionextractedbehaviour_maniset617be95b__scimitar_horned_oryx_male_standgimmick01_59.npy` cap1 (12w/72c): 'The male horned oryx stands still using his gimmick to remain motionless'
- `val` `animo4d_L4_safe` `PZ_Takin_Male_takin_male__animationnotmotionextractedfighting_maniset4360f71__takin_male_standgimmick01_316.npy` cap1 (10w/54c): 'The male takin stands still pretending to be a gimmick'
- `val` `animo4d_L4_safe` `PZ_Tasmanian_Devil_Female_tasmanian_devil_female__animationnotmotionextractedbehaviour_maniset2c94a132__tasmanian_devil_female_standgimmick_163.npy` cap1 (15w/98c): 'The female Tasmanian devil stands still pretending to be a gimmick to distract potential predators'
- `train` `truebones` `Centipede___Take_001_200.npy` cap0 (8w/47c): 'A centipede performs a short take 001 movement.'
- `train` `truebones` `Centipede___Take_001_200.npy` cap3 (8w/44c): 'A animal performs a short take 001 movement.'
- `train` `truebones` `Centipede___Take_001_200.npy` cap4 (9w/58c): 'A many-legged creature performs a short take 001 movement.'
- `train` `truebones` `Centipede___Take_001_200.npy` cap5 (8w/47c): 'A arthropod performs a short take 001 movement.'
- `train` `truebones` `Elephant___Take_001_315.npy` cap0 (8w/47c): 'An elephant performs a short take 001 movement.'
- `train` `truebones` `Elephant___Take_001_315.npy` cap3 (8w/45c): 'An animal performs a short take 001 movement.'
- `train` `truebones` `Elephant___Take_001_315.npy` cap4 (8w/45c): 'An mammal performs a short take 001 movement.'
- `train` `truebones` `Elephant___Take_001_315.npy` cap5 (8w/48c): 'An quadruped performs a short take 001 movement.'
- `train` `truebones` `HermitCrab___Take_001_411.npy` cap0 (9w/49c): 'A hermit crab performs a short take 001 movement.'
- `train` `truebones` `HermitCrab___Take_001_411.npy` cap3 (8w/44c): 'A animal performs a short take 001 movement.'
- `train` `truebones` `HermitCrab___Take_001_411.npy` cap4 (8w/48c): 'A crustacean performs a short take 001 movement.'
- `train` `truebones` `HermitCrab___Take_001_411.npy` cap5 (9w/59c): 'A multi-legged creature performs a short take 001 movement.'
- `train` `animo4d_L4_safe` `PZ_Aardvark_Female_aardvark_female__animationnotmotionextractedbehaviour_maniset1b75d874__aardvark_female_standgimmick01_67.npy` cap0 (10w/55c): 'The female aardvark stands still as part of her gimmick'
- `train` `animo4d_L4_safe` `PZ_Aardvark_Female_aardvark_female__animationnotmotionextractedbehaviour_maniset1b75d874__aardvark_female_standgimmick01_67.npy` cap1 (15w/84c): 'The female aardvark stands still then pretends to stand still as part of her gimmick'
- `train` `animo4d_L4_safe` `PZ_Aardvark_Juvenile_aardvark_juvenile__animationnotmotionextractedbehaviour_maniset828910c__aardvark_juvenile_standgimmick01_63.npy` cap1 (14w/86c): 'The juvenile aardvark stands still pretending to be a gimmick to distract its predator'
- `train` `animo4d_L4_safe` `PZ_Aardvark_Male_aardvark_male__animationnotmotionextractedbehaviour_maniset32599729__aardvark_male_standgimmick01_71.npy` cap0 (10w/53c): 'The male aardvark stands still as part of its gimmick'
- `train` `animo4d_L4_safe` `PZ_Addax_Juvenile_addax_juvenile__animationnotmotionextractedbehaviour_maniset1f0887ae__addax_juvenile_standgimmick01_64.npy` cap1 (10w/58c): 'The juvenile addax stands still pretending to be a gimmick'

