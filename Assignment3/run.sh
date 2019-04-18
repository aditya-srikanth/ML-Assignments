mkdir results

echo "Running the preprocessor"
python3 preproc.py 

echo "Running the neural net"
python3 driver.py
