# ChartInsight
Create a simple web app that allows users to upload a chart, ask questions about it, and receive answers from an integrated Large Language Model (LLM).

## Dataset  

Dataset used: [DeepRuleDataset](https://huggingface.co/datasets/niups/DeepRuleDataset/tree/main).  

### Instructions for Downloading and Processing the Data  

1. Download the dataset from the link above.  
2. Save the data in the `data` folder inside `assets`.  
3. Extract the downloaded files and move all extracted folders into the `data` directory.  
4. Run the `reduce_data.ipynb` notebook in the same folder to process and reduce the dataset size.  

## References
This work is based on:  

Liu, Xiaoyi, Klabjan, Diego, & Bless, Patrick N. (2019). *Data Extraction from Charts via Single Deep Neural Network*. arXiv preprint arXiv:1906.11906. [https://doi.org/10.48550/arXiv.1906.11906](https://doi.org/10.48550/arXiv.1906.11906).  
