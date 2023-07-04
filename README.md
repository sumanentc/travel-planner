## RoamWise Travel Planner

RoamWise is a travel planning application designed to enhance your exploration and make your journeys more seamless. With RoamWise, you can unlock the full potential of your travels by providing intelligent recommendations, comprehensive itineraries, and convenient travel management tools.

Discover new destinations and hidden gems tailored to your preferences, interests, and travel style. RoamWise combines advanced algorithms with curated travel content to offer personalized recommendations that suit your individual tastes. Whether you're an adventure seeker, a food enthusiast, a history buff, or simply looking to unwind, RoamWise ensures your travel experiences align with your desires.

Key features RoamWise include:
* Destination Recommendation: Users can discover perfect travel destination based on their preferences and interests.
* Itinerary Creation: Users can create detailed itineraries by inputting their destination, travel dates, and duration of the trip.
* Context based Recommendation: Users will get personalized travel recommendations tailored to their specific context and preferences.

Overall, the travel planner app aims to simplify the process of itinerary creation, enhance travel organization, and provide users with a seamless and enjoyable travel planning experience.

### Built With

- [Rasa : 3.5.2 ](https://rasa.com/docs/rasa/)
- [Python : 3.9 ](https://www.python.org/)
- [Rasa-SDK Action Server : 3.5.1 ](https://rasa.com/docs/action-server)
- [RASA-X :1.0.0](https://rasa.com/docs/rasa-x/)
- [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
- [ChatGPT](https://openai.com/blog/chatgpt)

## Getting Started

### Prerequisites

- Python
- [Pipenv](https://pypi.org/project/pipenv/)
- [Docker](https://docs.docker.com/engine/install/)
- [Helm](https://helm.sh/docs/intro/install/)
- [Kubernetes](https://kubernetes.io/docs/setup/)

### Installation

- Clone the repository

  ```
  git clone https://github.com/sumanentc/travel-planner.git
  ```

- Using RASA Shell and Stand alone Action Server

1. Install dependencies

  ```
  pipenv shell

  pipenv install
  ```

2. Train the model

  ```
  rasa train

  ```

3. Start the Action Server

  ```
  rasa run actions -vv

  ```

4. Start the RASA shell

  ```
  rasa shell -v
  ```

5. Start asking questions on the RASA shell

- Using Docker Compose for Installation
  **Note** : Here I am using my personal docker hub account to store the image: **sumand**

- Install the Bot along with RASA-X UI

1. Build Action Server Docker image

```
docker build actions/ -t sumand/rasa-action-server:3.5.1

docker push sumand/rasa-action-server:3.5.1

```

2. Build Rasa NLU Docker image

```
docker build . -t sumand/rasa-server:3.5.2

docker push sumand/rasa-server:3.5.2
```

3. Install RASA-X. I used [Helm-Chart](https://rasa.com/docs/rasa-x/installation-and-setup/install/helm-chart) for installation.

3.1 Create new namespace for rasa deployment

```
kubectl create namespace rasa
```

3.2 deploy RASA-X using the Helm Chart along with the customization specified in values.yml

```
helm repo add rasa-x https://rasahq.github.io/rasa-x-helm

helm --namespace rasa install --values values.yml my-release rasa-x/rasa-x
```

3.3 Update the Helm Chart in case we need any changes

```
helm --namespace rasa upgrade --values values.yml my-release rasa-x/rasa-x
```
3.4 Delete all the deployment in case not required

```
helm uninstall my-release -n rasa

```

4. Deploy [RASA-X](https://rasa.com/docs/rasa-x/installation-and-setup/deploy)

After executing the above Helm Chart, check RAXA-X is deployed successfully. Execute the below commands to check if all the pods are up and running. Else check the logs of individual pods for ERROR.

```
kubectl get pods -n rasa
```

Once all the pods are up and running then the RASA-X UI can be opened using the below url. Use the Password specified in Values.yml file to login.

```
http://localhost:8000/login
```

![RASA-X ](./images/RASA-X-Login.png)

Upload the model after login and make the model active

![RASA-X ](./images/upload-model.png)

## License

Distributed under the MIT License. See `LICENSE` for more information.