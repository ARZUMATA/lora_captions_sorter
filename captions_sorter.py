import os
import json
import torch
from pathlib import Path
from transformers import CLIPTokenizer
from collections import Counter
from typing import List, Dict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

tag_group_order = [
    "keep_tokens",
    "primary_features",
    "hair_features",
    "accessories",
    "body_features",
    "poses",
    "makuep",
    "position_view",
    "clothing",
    "style",
    "unsorted",
    "background",
    ]

keep_first_tags = 1

class TagFileClass:
    def __init__(self, file_name: str, full_path: str, raw_tags: List[str]):
        self.file_name = file_name
        self.raw_tags = raw_tags
        self.full_path = full_path
        self.tag_groups: List[TagsGroupClass] = []

    def cleanup_banned_tags(self):
        for raw_tag in self.raw_tags:
            if raw_tag in banned_tags:
                self.raw_tags.remove(raw_tag)
        
    def cleanup_empty_tags(self):
        self.raw_tags = [tag for tag in self.raw_tags if tag.strip()]

    def arrange_tags_in_groups(self):
        raw_tags_copy = self.raw_tags.copy()
        
        if keep_first_tags > 0:
            # Initialize an empty list to store the selected tags
            kept_tokens = []

            # Move 'X' tokens from raw_tags_copy to kept_tokens
            asd = raw_tags_copy[:keep_first_tags]
            for tag in raw_tags_copy[:keep_first_tags]:
                kept_tokens.append(tag)
                raw_tags_copy.remove(tag)

            keep_group = next((group for group in tag_groups if group.name == "keep_tokens"), None)
            self.tag_groups.append(TagsGroupClass(keep_group.priority, keep_group.name, kept_tokens))

        # Create groups for each tag group and add matching tags
        for group in tag_groups:
            matching_tags = []
            for tag in raw_tags_copy[:]:  # Iterate over a copy to safely modify
                if tag in group.tags:
                    matching_tags.append(tag)
                    raw_tags_copy.remove(tag)
            
            if matching_tags:  # Only add group if it has matching tags
                self.tag_groups.append(TagsGroupClass(group.priority, group.name, matching_tags))
        
        # Add remaining unmatched tags to unsorted group
        if raw_tags_copy:
            #Get unsorted group
            unsorted_group = next((group for group in tag_groups if group.name == "unsorted"), None)
            # self.tag_groups.append(TagsGroupClass(999, "unsorted", raw_tags_copy))
            self.tag_groups.append(TagsGroupClass(unsorted_group.priority, unsorted_group.name, raw_tags_copy))

    def list_unsorted_tags(self):
        unsorted_tags = []
        if self.tag_groups:
            for group in self.tag_groups:
                if group.name == "unsorted":
                    print(f"Unsorted tags in {self.file_name}:")
                    for tag in group.tags:
                        unsorted_tags.append(tag)
        return unsorted_tags
    
    def save_tags_to_txt(self):
        # Sort the tag groups by priority
        sorted_tag_groups = sorted(self.tag_groups, key=lambda x: x.priority)

        # Aggregate all tags from all tag groups into a single list
        all_tags = []
        for tag_group in sorted_tag_groups:
            all_tags.extend(tag_group.tags)

        # Save the aggregated tags back to the file
        with open(self.full_path, 'w', encoding='utf-8') as f:
            # Join all tags together with commas
            tags_str = ', '.join(all_tags)
            # Write the concatenated tags to the file
            f.write(tags_str)

class TagsGroupClass:
    def __init__(self, priority: int, name: str, tags: List[str]):
        self.priority = priority
        self.name = name
        self.tags = tags

    def sort_tags(self):
        print(f"\tSorting tags in group: {self.name}")
        self.tags = self.sort_tags_by_token_length(self.tags)

    def sort_tags_by_token_length(self, tags, ascending=False):
        # Create list of (tag, token_length) tuples
        tag_lengths = []

        for tag in tags:
            # Check cache first
            if tag in token_length_cache:
                token_length = token_length_cache[tag]
            else:
                # Calculate and cache if not found
                tokens = tokenizer(tag)['input_ids']
                token_length = len(tokens)
                token_length_cache[tag] = token_length

            tag_lengths.append((tag, token_length))
        
        # Sort by token length
        sorted_tags = sorted(tag_lengths, key=lambda x: x[1], reverse=not ascending)
        
        # Return just the sorted tags
        return [tag for tag, _ in sorted_tags]

# Global variables
tag_files: List[TagFileClass] = []
tag_groups: List[TagsGroupClass] = []
banned_tags: List[str] = []
unsorted_tags: List[str] = []
total_tags: List[str] = []
token_length_cache: Dict[str, int] = {}

# Initialize CLIP tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def load_tag_files(tag_files_dir):
    tag_files_path = Path(tag_files_dir)
    tag_files = []

    def recursive_load(directory):
        if directory.exists():
            # Process all txt files in current directory
            for file in directory.glob('*.txt'):
                with open(file, 'r', encoding='utf-8') as f:
                    raw_tags_csv = [line.strip() for line in f.readlines() if line.strip()]

                    # Split comma separated line to indidual lines
                    raw_tags = [tag.strip() for tag in raw_tags_csv[0].split(',')]

                    full_path = file.resolve()
                    tag_files.append(TagFileClass(file.stem, full_path, raw_tags))
            
            # Recursively process all subdirectories
            for subdir in directory.iterdir():
                if subdir.is_dir():
                    recursive_load(subdir)

    recursive_load(tag_files_path)
    return tag_files

def load_banned_tags():
    banned_tags_path = Path(".\\banned_tags.txt")
    banned_tags = []

    if banned_tags_path.exists():
        with open(banned_tags_path, 'r', encoding='utf-8') as f:
            banned_tags = [line.strip() for line in f.readlines() if line.strip()]

    return banned_tags

def load_group_tags():
    groups_dir = ".\groups"
    groups_path = Path(groups_dir)
    
    if groups_path.exists():
        tag_groups = []
        for file in groups_path.glob('*.txt'):
            group_name = file.stem

            try:
                priority = tag_group_order.index(group_name)
                print(f"Group: {group_name}, Priority: {priority}")
            except ValueError:
                priority = 999
                print(f"Group: {group_name} is not in the tag group order list.")
            
            with open(file, 'r', encoding='utf-8') as f:
                tags = [line.strip() for line in f.readlines() if line.strip()]

                # Create a new TagsGroupClass instance
                group = TagsGroupClass(priority, group_name, tags)
                # Add the group to the list
                tag_groups.append(group)
    return tag_groups

def sort_tag_groups(tag_file):
    for group in tag_file.tag_groups:
        group.sort_tags()

if __name__ == "__main__":
    root_directory = "your/path/to/tags"  # Replace with your directory path

    # Load tag files
    print("Loading tag files...")
    tag_files = load_tag_files(root_directory)

    # Load banned tags
    print("Loading banned tags...")
    banned_tags = load_banned_tags()

    # Load group tags
    print("Loading group tags...")
    tag_groups = load_group_tags()
    
    # Cleanup empty tags
    print("Cleaning up empty tags...")
    for tag_file in tag_files:
        tag_file.cleanup_empty_tags()

    # Cleanup banned tags
    print("Cleaning up banned tags...")
    for tag_file in tag_files:
        tag_file.cleanup_banned_tags()

    # Arrange raw tags in groups
    print("Arranging tags in groups...")
    for tag_file in tag_files:
        tag_file.arrange_tags_in_groups()

    # Sort each tag group using multiple threads
    print("Sorting tags in groups...")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(sort_tag_groups, tag_files)

    # Get total tags
    print("Getting total tags...")
    for tag_file in tag_files:
        for tag in tag_file.raw_tags:
            if tag not in total_tags:
                total_tags.append(tag)

    #  Report unsorted tags
    print(" Report unsorted tags...")
    for tag_file in tag_files:
        tag_file_unsorted_tags = tag_file.list_unsorted_tags()
        for tag in tag_file_unsorted_tags:
            if tag not in unsorted_tags:
                unsorted_tags.append(tag)
    print("Unsorted tags:")
    print(unsorted_tags)

    # Print tags per group
    print("Printing tags per group...")
    for tag_file in tag_files:
        print(f"TagFile: {tag_file.file_name}")
        for tag_group in tag_file.tag_groups:
            print(f"Group: {tag_group.name}, Priority: {tag_group.priority}")
            for tag in tag_group.tags:
                print(tag)

    # Save tags back
    print("Saving tags back...")
    for tag_file in tag_files:
        tag_file.save_tags_to_txt()
    

    print("Done!")
