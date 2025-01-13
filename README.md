### Project description

#### Air Quality Prediction

This project aims to develop a machine learning model to predict air quality based on various environmental factors. The dataset used for this project is updated_pollution_dataset.csv, which contains measurements of different pollutants and environmental conditions along with the corresponding air quality index.

### Dataset
The dataset consists of the following columns:

* `Temperature`: The ambient temperature in degrees Celsius.
* `Humidity`: The relative humidity percentage.
* `PM2.5`: The concentration of particulate matter less than 2.5 micrometers in micrograms per cubic meter.
* `PM10`: The concentration of particulate matter less than 10 micrometers in micrograms per cubic meter.
* `NO2`: The concentration of nitrogen dioxide in micrograms per cubic meter.
* `SO2`: The concentration of sulfur dioxide in micrograms per cubic meter.
* `CO`: The concentration of carbon monoxide in milligrams per cubic meter.
* `Proximity_to_Industrial_Areas`: The proximity to industrial areas in kilometers.
* `Population_Density`: The population density in people per square kilometer.
* `Air Quality`: The air quality index categorized as Good, Moderate, Poor, or Hazardous.

### Setting Up

#### Clone the repository
```
git clone https://github.com/Jaykold/Air-quality-classifier.git
```
#### Navigate to the project directory
`cd air-quality-classifier`

#### Create a virtual environment
`python -m venv myenv`

After creating the virtual environment, you need to activate it:

* On Windows:
`myenv\Scripts\activate`

* On macOS and Linux:
`source myenv/bin/activate`

You can install the project dependencies by running this command:

`pip install -r requirements.txt`

#### Install the required dependencies
```
pip install -r requirements.txt
```

### Usage

#### Training the model
To train the model, run the following command:
```
python src/model_trainer.py
```

#### Making predictions
To make predictions using the trained model, run:
```
python predict.py
```

#### Running the Application
To run the application, execute the following command:

```
python -m main or python main.py
```

#### Building & Running the Docker container
1. Build the docker image using this code
```
docker build -t air-quality-classifier .
```

2. Run the docker container using this code
```
docker run -it --rm -p 8000:8000 air-quality-classifier
```

3. Test the running container
```
python test.py
```

4. Push to DockerHub (Optional)
```
docker tag air-quality-classifier <dockerhubName/air-quality-classifier:latest>
```