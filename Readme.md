# BBC News Classification Project

Welcome to the BBC News Classification project! This repository contains all the code and resources required to build and deploy a news classification system that categorizes BBC news articles into Business, Tech, Sport, Politics, and Entertainment categories using Natural Language Processing (NLP) techniques and Non-Negative Matrix Factorization (NMF).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data](#data)
4. [Preprocessing](#preprocessing)
5. [Model Building](#model-building)
6. [Deployment](#deployment)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

## Project Overview
The aim of this project is to develop an efficient and accurate text classification system to categorize BBC news articles into predefined categories. We leverage Non-Negative Matrix Factorization (NMF) for topic modeling and deploy the model using Gradio on Hugging Face for easy access and interaction.

## Installation
To run this project, you'll need to have Python installed on your machine. Follow the steps below to set up the environment:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/BBC-News-Classification.git
    cd BBC-News-Classification
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Data
The dataset used in this project is the BBC News dataset. It contains news articles categorized into Business, Tech, Sport, Politics, and Entertainment. Ensure that the dataset is stored in the `dataset/` directory.

## Preprocessing
Preprocessing steps include:
- Tokenization
- Removing stopwords
- Lemmatization
- Vectorization using TF-IDF

Run the preprocessing script:
```sh
python preprocessing.py
```

## Model Building
We use Non-Negative Matrix Factorization (NMF) for topic modeling and classification. The model building script includes training and evaluation of the model.

## Deployment
The model is deployed using Gradio on Hugging Face, providing an interactive web interface for users to classify new articles.

To deploy the model locally:
```sh
python app.py
```

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact me at [amansrivastava7969@gmail.com](mailto:amansrivastava7969@gmail.com) or connect with me on [LinkedIn Post](https://www.linkedin.com/posts/aman-srivastava-b60594245_internshipjourney-aiinnovation-nlptechniques-activity-7224015558323662848-KnLT?utm_source=share&utm_medium=member_desktop).

---

Thank you for checking out the BBC News Classification project! If you found this project helpful, please consider starring the repository. Happy coding! 🚀
