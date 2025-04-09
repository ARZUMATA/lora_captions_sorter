# LoRa Tags Sorter

A comprehensive tool for organizing and sorting tags in LoRa training datasets.

## Overview

This tool helps manage and organize tags (captions) for LoRa model training by:

- Processing text files containing comma-separated tags from directories recursively
- Cleaning up empty and banned tags
- Organizing tags into predefined priority groups
- Sorting tags within groups by token length
- Removing tags that appear less than a specified threshold
- Saving the sorted tags back to the original files

## Features

- **Hierarchical Tag Organization**: Sorts tags according to predefined group priorities
- **Token-Length Sorting**: Within each group, sorts tags by CLIP token length for optimal prompt organization
- **Banned Tags Filtering**: Removes unwanted tags from a customizable banned list
- **Low-frequency Tag Removal**: Option to remove tags that appear less than a specified threshold
- **Duplicate Group Detection**: Identifies and reports tags that appear in multiple groups
- **Multithreaded Processing**: Uses thread pools for efficient tag sorting
- **Format Conversion**: Includes utilities to convert between TXT and JSON formats

## Tag Group Priority System

Tags are organized into the following priority groups (from highest to lowest):

1. `keep_tokens` - Essential tags that should always appear first
2. `primary_features` - Key identifying features
3. `hair_features` - Hair-related descriptors
4. `accessories` - Items worn or carried
5. `body_features` - Physical characteristics
6. `poses` - Body positioning
7. `makeup` - Cosmetic elements
8. `position_view` - Camera angles and perspectives
9. `clothing` - Attire descriptions
10. `actions` - What the subject is doing
11. `other` - Miscellaneous descriptors
12. `style` - Artistic style elements
13. `nsfw` - Adult content markers
14. `unsorted` - Tags not assigned to any group
15. `background` - Setting and environment descriptors

## Tag Ordering Philosophy

Tag ordering matters for optimal results:

- Most important/specific tags first
- Style elements next
- Colors and materials
- Positioning/angles
- Generic descriptors last

This priority system ensures that the most distinctive and important features are emphasized in the model training process.

## Usage

1. Create a `groups` directory with text files named after each group (e.g., `primary_features.txt`, `style.txt`)
2. Add relevant tags to each group file, one tag per line
3. Create a `banned_tags.txt` file with tags to exclude (optional)
4. Modify the `root_directory` in the script to point to your dataset
5. Adjust `tag_count_threshold` if needed (default: 5)
6. Run the script:

```bash
python captions_sorter.py
```

7. Review the unsorted tags (if any) and add them to appropriate group files
8. Confirm to save changes when prompted

## Directory Structure

```
/
├── captions_sorter.py        # Main script
├── banned_tags.txt           # Tags to exclude (optional)
├── groups/                   # Tag group definitions
│   ├── keep_tokens.txt
│   ├── primary_features.txt
│   ├── style.txt
│   └── ...
└── your_dataset_directory/   # Directory containing tag files to process
```

## Configuration Options

- `keep_first_tags`: Number of tags to always keep at the beginning (default: 0)
- `tag_count_threshold`: Minimum number of occurrences required to keep a tag (default: 5)

## Process Flow

1. Load all tag files from the specified directory recursively
2. Load banned tags and group definitions
3. Check for duplicate tags across groups
4. Clean up empty and banned tags
5. Arrange tags into their respective groups
6. Sort tags within each group by token length
7. Report any unsorted tags
8. Count tag occurrences across all files
9. Remove tags that appear less than the threshold
10. Save the sorted tags back to the original files (after confirmation)

## Requirements

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)

## License

MIT License - See LICENSE.md for details

## Notes

- The script will report any tags that appear in multiple group files to avoid conflicts
- Unsorted tags will be reported at the end of processing for manual categorization
- The script will ask for confirmation before saving changes to files
- Compatible with [dataset-tag-editor-standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone)
