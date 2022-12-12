<h1 align="center">Hi 👋, we are a ComSec Lab</h1>
<h3 align="center">A security specialists from St. Petersburg</h3>

- 🔭 We are currently working on **AOPSSOP Library**

<h3 align="left">Library</h3>
<p align="left">
The developed library fulfills 3 main tasks:

- **Preprocessing** of raw and fuzzy data.
- **Assessment** of the current state of complex technical objects.
- **Forecasting** of the future state of complex technical objects.

Let's consider each task and its modules in more detail.
</p>

<h3 align="left">Preprocessing</h3>
<p align="left">
This task is fulfilled with the help of the following modules:

- POSND: .
- IZSND: .

Examples that are describing the work with POSND module are presented in examples/posnd_examples.py.

1. TBD

Examples that are describing the work with IZSND module are presented in examples/izsnd_examples.py.

1. TBD

</p>

<h3 align="left">Assessment</h3>
<p align="left">
This task is fulfilled with the help of the AOSSOP module. Its work consists in applying a pre-trained machine learning model that allows you to determine the state of an object from its descriptive attributes. Conventionally, the algorithm can be presented in the form of three stages:

1. Extracting the attributes of the analyzed object.
2. Normalizing the attributes.
3. Assessing the state of the object.

Within the framework of the tasks considered in the project, the first stage involves reading attributes from a text file with parsing of the corresponding fields. At the second stage, the fields are preprocessed in order to bring them to a single interval. Finally, at the third stage, a pre-trained neural network is launched, which determines whether an object belongs to a particular state based on its attributes.

Examples that are describing the work with AOSSOP module are presented in examples/aossop_examples.py.

Examples of the application of this algorithm cover the task of ensuring the cybersecurity of critical resources and objects, as well as the task of determining the trajectory of a vehicle (crane). In the first case, the data obtained from the sensors of the system of steam turbines and pumped storage power plants are considered as input data. In the second case, the input data are parameters that describe the operation and movement of the overhead crane under various loads.

The essence of the experiment was to test the suitability of a pre-configured model as part of the task of assessing the state of a critically important object. During the experiment, two phases were distinguished: the training phase and the testing phase. At the first phase, the weights of the neural network were adjusted, and at the second phase, the calculation of performance indicators for estimating the state of the analyzed object was carried out.
</p>

<h3 align="left">Forecasting</h3>
<p align="left">
This task is fulfilled with the help of the APSSOP module. It performs:

- training a model for forecasting the states of complex systems based on historical data in the form of a time series, which is a sequence of feature vectors of system states;
- autonomous forecasting of system states described by a certain feature vector for a given period of time.

To train the model, the length and sequence of features must be constant for the state of the system at any given time. The model predicts only the numerical parameters of the system states.

Examples that are describing the work with APSSOP module are presented in ```examples/apssop_examples.py```.

1. Function ```example_appsop_model_training(dataset_name, suf='', mode=1)``` – example of training prediction and data normalization models (```dataset_name```: name of dataset (str), ```suf```: suffix for naming the output (str), ```mode```: boot mode, for developers (integer)).
2. Function ```example_appsop_forecasting(dataset_name, suf='', mode=1, independently=True, sample_type='test')``` - example of data forecasting based on an existing model, including predictive estimation (```dataset_name```: name of dataset (str), ```suf```: suffix for naming the output (str), ```mode```: boot mode, for developers (integer), ```independently```: sequence is predicted depending on past values or not (boolean), ```sample_type```: type of forecasting sample for estimation - train or test (str)).

If param ```independently``` is ```True``` then all feature value vectors are predicted independently of each other. At each forecasting stage, an element of the target sample is added to the batch. If param ```independently``` is ```False``` then each predicted vector becomes an element of a new package for subsequent forecasting.

If param ```sample_type``` is ```'train'``` then forecasting time window is equal to the length of training sample from the second to the last batch. Batch for forecasting is the first batch of the training sample. True values for estimation is values of the training sample from the second to the last batch. If param ```sample_type``` is ```'test'``` then forecasting time window is equal to the length of all test sample. Batch for forecasting is the last batch of the training sample. True values is values of the test sample.
</p>



<h3 align="left">Connect with us:</h3>
<p align="left">labcomsec@gmail.com
</p>

<p align="left">
The project is developed and maintained by the research team of the Laboratory of Computer Security Problems (SPC RAS), 
which is a part of the Research Center "Strong Artificial Intelligence in Industry" (ITMO University).
</p>

<p align="left">
<a href="https://sai.itmo.ru/" target="_blank" rel="noreferrer">
<img src="https://sai.itmo.ru/img/subdomains/sai/logo.png" alt="Сильный ИИ" width="184" height="50"/></a></p>

<h3 align="left">Languages and Tools:</h3>
<p align="left">
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer">
<img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/>
</a>
<a href="https://www.python.org" target="_blank" rel="noreferrer">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/>
</a>
<a href="https://scikit-learn.org/" target="_blank" rel="noreferrer">
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/>
</a>
<a href="https://www.tensorflow.org" target="_blank" rel="noreferrer">
<img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/>
</a>
</p>
