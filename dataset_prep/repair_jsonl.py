import json
import os

def repair_jsonl(path):
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    with open(path, 'r') as f:
        content = f.read()

    # The file has literal '\n' characters (backslash and 'n') instead of actual newlines.
    # We want to split the content into individual JSON objects.
    # Since each object starts with '{' and ends with '}', and they are separated by '\n' (literal),
    # we can use json.JSONDecoder.raw_decode to pull them out one by one.

    decoder = json.JSONDecoder()
    pos = 0
    objs = []
    
    print(f"Repairing {path} (Size: {len(content)} bytes)...")
    
    while pos < len(content):
        # Skip any whitespace or the literal '\n' characters
        while pos < len(content) and content[pos] not in '{':
            pos += 1
        
        if pos >= len(content):
            break
            
        try:
            obj, pos = decoder.raw_decode(content, pos)
            objs.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error decoding at position {pos}: {e}")
            # Try to find the next '{' to resume
            pos += 1
            
    print(f"Found {len(objs)} objects.")
    
    with open(path, 'w') as f:
        for obj in objs:
            f.write(json.dumps(obj) + '\n')
    
    print("Repair complete.")

if __name__ == "__main__":
    repair_jsonl('/Users/diwanshuyadav/Documents/mlops_proj/dataset.jsonl')
