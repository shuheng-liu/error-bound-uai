# Dependencies

Install all dependencies with
```bash
pip3 install -r requirements.txt
```

# Run

Run the scripts with 
```bash
python3 ode_2nd_order.py
python3 ode_system.py
python3 duffing.py
python3 pde_attractor.py
python3 pde_unsolvable.py
```

# Output
Each of the above scripts will generate a plot (.pdf) in an `assets` folder, which will be a sibling of the `code` folder. 

# Debug
If an error occurs, chances are it's an issue with matplotlib's plotting backend.
Make sure you have TexLive and Computer Modern Font installed locally. Otherwise, the plotting might fail.