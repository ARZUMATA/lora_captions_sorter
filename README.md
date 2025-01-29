# LoRa Tags sorter

Simple tag sorter for LoRa's.

* Get tags from txt files from directories recursively.
* Remove tags we don't want
* Sort using predefined group order (in code) where group tags exist in groups directory.
* Then sort using token length descending.

keep_tokens - if required, always first.

unsorted - every undefined tag goes here and we can put it anywhere we want.

### Idea:
We have a rule-set where we prioritize some tags over others

Tag ordering matters:

* Most important/specific tags first
* Style elements next
* Colors and materials
* Positioning/angles
* Generic descriptors last

So we do groups sorting (from highest to lowest priority):

* Key Feature Tokens
* Style/Material Tokens
* Color/Pattern Tokens
* Position/Angle Tokens
* Background/Generic Tokens

Within each priority group, we sort by token length.

I'm using [dataset-tag-editor-standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone) but it lacks that feature or rule-set to put tags in groups with predefined order.
