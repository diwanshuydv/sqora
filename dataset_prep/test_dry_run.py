import json
from dataset_pipeline import generate_descriptions, generate_manim_code

def main():
    print("Testing generation of descriptions and manim code...")
    subtopic = "Bohr's model of hydrogen atom and spectral lines"
    
    desc_list = generate_descriptions(subtopic, num_desc=1)
    if not desc_list:
        print("Failed to generate description.")
        return
        
    desc = desc_list[0]
    print(f"\nGenerated Description: {desc}")
    
    print("\nGenerating Manim code...")
    code = generate_manim_code(desc)
    print("\nGenerated Manim Code:\n")
    print(code)

if __name__ == "__main__":
    main()
