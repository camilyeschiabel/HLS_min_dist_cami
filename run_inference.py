import sys
import os
import subprocess

def run_inference(prompt):
    # Tokenize input using reference Microsoft/BitNet tokenizer script
    #sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '../BitNet/gpu')))
    from tokenizer import Tokenizer
    tok = Tokenizer(os.path.normpath(os.path.join(os.path.dirname(__file__), './tokenizer.model')))
    
    # Extract IDs corresponding to precisely the reference pipeline
    prompt_ids = tok.encode(prompt, bos=True, eos=False)
    
    # Format list of IDs as space separated string for C testbench
    token_str = " ".join(str(tid) for tid in prompt_ids)
    
    tcl_content = f"""open_project bitnet_hls_proj_sim -reset
set_top bitnet_inference
add_files bitnet.c -cflags "-std=c99"
add_files -tb testbench.c -cflags "-std=c99"
open_solution "solution1" -reset
set_part {{xcvu9p-flga2104-2-i}}
create_clock -period 10 -name default
csim_design -argv "{token_str}"
exit
"""
    with open("run_inference.tcl", "w") as f:
        f.write(tcl_content)
        
    print(f"Starting native HLS C-Simulation with prompt: '{prompt}'...")
    print(f"Tokenized context natively matching reference: {prompt_ids}")
    print("This will compile the mathematically accurate C kernels using Clang.", flush=True)
    
    cmd = 'vitis-run --mode hls --tcl run_inference.tcl'
    
    try:
        # Instead of os.system, capture stdout directly to decode [OUT] tokens
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True)
        
        print("\n==========================================")
        print(" BitNet b1.58 (2B-4T) HLS Simulation")
        print("==========================================\n")
        print(f"{prompt}", end="", flush=True)
        
        for line in process.stdout:
            # Check for our numeric outputs
            if "[OUT]" in line:
                token_val = int(line.strip().split("[OUT]")[1])
                word = tok.decode([token_val])
                print(word, end="", flush=True)
            elif "Simulation Complete" in line:
                print("\n\n[Simulation Complete]")
            else:
                print(line.strip(), flush=True)
                
        process.wait()
        
    except Exception as e:
        print(f"Failed to execute Vitis: {e}")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "How much is 1+1?"
    run_inference(prompt)
