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
    "makeup",
    "position_view",
    "clothing",
    "actions",
    "other",
    "style",
    "nsfw",
    "unsorted",
    "background",
    ]

keep_first_tags = 0
tag_count_threshold = 5

class TagFileClass:
    def __init__(self, file_name: str, full_path: str, raw_tags: List[str]):
        self.file_name = file_name
        self.raw_tags = raw_tags
        self.full_path = full_path
        self.tag_groups: List[TagsGroupClass] = []

    def cleanup_banned_tags(self) -> List[str]:
        cleaned_tags = [tag for tag in self.raw_tags if tag not in banned_tags]
        removed_tags = set(self.raw_tags) - set(cleaned_tags)
        return list(removed_tags)
        
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

def report_duplicate_group_tags():
    # Create a dictionary to store tags and their groups
    tag_to_groups = {}
    
    # Iterate through all groups and their tags
    for group in tag_groups:
        for tag in group.tags:
            if tag not in tag_to_groups:
                tag_to_groups[tag] = []
            tag_to_groups[tag].append(group.name)
    
    # Find and report tags that appear in multiple groups
    duplicates_found = False
    for tag, groups in tag_to_groups.items():
        if len(groups) > 1:
            duplicates_found = True
            print(f"Tag '{tag}' found in multiple groups: {', '.join(groups)}")
    
    if duplicates_found:
        print("Please fix duplicate tags before continuing.")
        exit()


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

def write_tags_and_count_to_txt(self, threshold=1):
    # Get all tags from all groups
    all_tags = []
    for group in self.tag_groups:
        all_tags.extend(group.tags)
    
    # Count occurrences of each tag
    tag_counts = Counter(all_tags)
    
    # Sort tags by count in descending order
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create output filename by adding '_counts' before extension
    output_path = Path(self.full_path).with_stem(Path(self.full_path).stem + '_counts')
    for tag, count in sorted_tags:
        print(f"{tag}: {count}\n")

    # Write counts to file, only including tags above threshold
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     for tag, count in sorted_tags:
    #         if count >= threshold:
    #             f.write(f"{tag}: {count}\n")

def get_total_tags_count(tag_files: List[TagFileClass]) -> Dict[str, int]:
    tag_counter = Counter()
    
    for tag_file in tag_files:
        # Get all tags from all groups in the file
        for group in tag_file.tag_groups:
            tag_counter.update(group.tags)
    
    # Sort by count in descending order
    sorted_tags = dict(sorted(tag_counter.items(), key=lambda x: x[1], reverse=True))
    return sorted_tags

def remove_tags_below_threshold(tag_files: List[TagFileClass], total_tags_count: Dict[str, int], threshold: int = 2) -> List[str]:
    removed_tags_set = set()  # Using a set to store unique removed tags
    for tag_file in tag_files:
        for group in tag_file.tag_groups:
            new_tags = [tag for tag in group.tags if total_tags_count[tag] >= threshold]
            removed_tags = set(group.tags) - set(new_tags)
            removed_tags_set.update(removed_tags)  # Add all tags from current file to the set
            group.tags = new_tags

    return list(removed_tags_set)


if __name__ == "__main__":
    root_directory = "S:\StableDiffusion_Data\Datasets\_Training_1_ToDo\AlterCiri_Striped_Socks_Goth_Dress_Pony\img\\10_alterciri"  # Replace with your directory path

    # Load tag files
    print("Loading tag files...")
    tag_files = load_tag_files(root_directory)

    # Load banned tags
    print("Loading banned tags...")
    banned_tags = load_banned_tags()

    # Load group tags
    print("Loading group tags...")
    tag_groups = load_group_tags()
    
    # Detect duplucate tags between groups
    print("Detecting duplicate tags between groups...")
    report_duplicate_group_tags()

    # Cleanup empty tags
    print("Cleaning up empty tags...")
    for tag_file in tag_files:
        tag_file.cleanup_empty_tags()

    # Cleanup banned tags
    print("Cleaning up banned tags...")
    removed_tags_set = set()  # Using a set to store unique removed tags
    for tag_file in tag_files:
        removed_tags_list = tag_file.cleanup_banned_tags()
        removed_tags_set.update(removed_tags_list)  # Add all tags from current file to the set
        
    # Convert set back to list and print as comma-separated string
    removed_tags_list = list(removed_tags_set)
    removed_tags_str = ', '.join(removed_tags_list)
    print("Removed tags (unique):", removed_tags_str)

    # Arrange raw tags in groups
    print("Arranging tags in groups...")
    for tag_file in tag_files:
        tag_file.arrange_tags_in_groups()

    # Sort each tag group using multiple threads
    print("Sorting tags in groups...")

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(sort_tag_groups, tag_files)

    #  Report unsorted tags
    print("Report unsorted tags...")
    for tag_file in tag_files:
        tag_file_unsorted_tags = tag_file.list_unsorted_tags()
        for tag in tag_file_unsorted_tags:
            if tag not in unsorted_tags:
                unsorted_tags.append(tag)

    # Print tags per group
    print("Printing tags per group...")
    for tag_file in tag_files:
        print(f"TagFile: {tag_file.file_name}")
        for tag_group in tag_file.tag_groups:
            print(f"\nGroup: {tag_group.name}, Priority: {tag_group.priority}")
            for tag in tag_group.tags:
                print(f"\t\t{tag}")

    # Get total tags
    print("Getting total tags...")
    for tag_file in tag_files:
        for tag in tag_file.raw_tags:
            if tag not in total_tags:
                total_tags.append(tag)
    print(f"Total tags: {', '.join(total_tags)}")

    # Count total tags across all files
    print("Counting total tags across all files...")
    total_tags_count = get_total_tags_count(tag_files)
    
    # Print results
    for tag, count in total_tags_count.items():
        print(f"{tag}: {count}")

    # Then in main, after total_tags_count is calculated:
    print("Removing tags below threshold...")
    removed_tags_list = remove_tags_below_threshold(tag_files, total_tags_count, threshold=tag_count_threshold)
    removed_tags_str = ', '.join(removed_tags_list)
    print("Removed tags (unique):", removed_tags_str)

    # Remove unsorted tags if below threshold
    print("Removing unsorted tags if below threshold...")
    cleaned_unsorted_tags = []
    for tag in unsorted_tags:
        if total_tags_count[tag] >= tag_count_threshold:
            print(f"Keeping unsorted tag: {tag} as count is {total_tags_count[tag]}")
            cleaned_unsorted_tags.append(tag)
        else:
            print(f"Removing unsorted tag: {tag} as count is {total_tags_count[tag]}")

    unsorted_tags = cleaned_unsorted_tags

    # Check for unsorted tags
    if (len(unsorted_tags) > 0):
        print("Unsorted tags found. Please sort them manually.")
        print("Unsorted tags:")
        for tag in unsorted_tags:
            print(tag)
        exit()


    # Save tags back
    response = input(f"Save changes? (y/N): ").strip().lower()
    if response == 'y':
        print("Saving tags back...")
        for tag_file in tag_files:
            tag_file.save_tags_to_txt()
    

    print("Done!")








def convert_txt_to_json(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_path = Path(dirpath) / filename
                json_path = txt_path.with_suffix('.json')
                
                with open(txt_path, 'r', encoding='utf-8') as f:
                    tags = f.read().strip()
                
                json_data = {
                    "version": "1.0",
                    "tags": {
                        "tags": tags
                    }
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=4, ensure_ascii=False)

def convert_json_to_txt(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                json_path = Path(dirpath) / filename
                txt_path = json_path.with_suffix('.txt')
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    tags = json_data.get('tags', {}).get('tags', '')
                
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(tags)






    # Convert txt to json
    # convert_txt_to_json(root_directory)
    
    # Convert json to txt
    # convert_json_to_txt(root_directory)



