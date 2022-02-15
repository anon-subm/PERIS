# PERIS

## Requirements
* Python
* Pytorch
* Python libararies in  <code> package-list.txt </code>

## How to run
1. Download data in ./data/ directory
    <pre><code> wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Cell_Phones_and_Accessories.csv </code></pre>
2. Split data into training/validation/test datasets
    <pre><code> python split_data.py ratings_Cell_Phones_and_Accessories.csv </code></pre>
3. Prepare data for training
    <pre><code> python build_recdata.py cell </code></pre>
4. Training
    <pre><code> python train.py --batch_size=128 --dataset=cell --lamb=0.7 --learning_rate=0.005 --model_name=peris --mu=0.5 --tau=0.5 </code></pre>

