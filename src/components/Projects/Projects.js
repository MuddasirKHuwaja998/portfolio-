import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import ProjectCard from "./ProjectCards";
import Particle from "../Particle";
import leaf from "../../Assets/Projects/leaf.png";
import emotion from "../../Assets/Projects/emotion.png";
import editor from "../../Assets/Projects/codeEditor.png";
import chatify from "../../Assets/Projects/chatify.png";
import suicide from "../../Assets/Projects/suicide.png";

function Projects() {
  return (
    <Container fluid className="project-section">
      <Particle />
      <Container>
        <h1 className="project-heading">
          My Recent <strong className="purple">Works </strong>
        </h1>
        <p style={{ color: "white" }}>
          Here are a few projects I've worked on recently.
        </p>
        <Row style={{ justifyContent: "center", paddingBottom: "10px" }}>
          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={require("../../Assets/PROJECT1.png")}
              isBlog={false}
              title="Otolaryngologists AI Diagnosis"
              description="This project is an AI-powered web application designed to assist in diagnosing ear-related diseases. It leverages machine learning models (CNN & MobileNetV3) for image-based disease detection and provides a user-friendly interface for users to upload and analyze medical images."
              ghLink="https://github.com/MuddasirKHuwaja998/OTOFARMA-AI-DIAGNOSIS.git"
              demoLink="https://chatify-49.web.app/" // keep demo link same, user will provide later
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={require("../../Assets/project2.png")}
              isBlog={false}
              title="AI Virtual Voice Assistant"
              description={"AI Virtual Voice Assistant:  It's a virtual voice assistant backend, integrating APIs for speech recognition, text-to-speech, email, and database management. It leverages NLP and cloud services to automate communication and appointment handling for users. A robust Flask app with advanced features like Google Cloud speech/text APIs, SendGrid email integration, and professional logging. Its value lies in combining AI, communication, and database management for scalable, production-grade applications."}
              ghLink="https://github.com/MuddasirKHuwaja998/Chatbot.git"
              demoLink="https://otobot-u589.onrender.com"
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={require("../../Assets/p[roject3.png")}
              isBlog={false}
              title={"Classifying & Detecting Emotions in Speech using CNN"}
              description={"A custom-built Convolutional Neural Network (CNN) architecture to classify six distinct emotional states from raw speech data. The primary technique involved converting audio waveforms into Mel-spectrograms, which represent time-frequency distributions and serve as the main input features for the model."}
              ghLink="https://github.com/MuddasirKHuwaja998/Classifying-Detecting-Emotions-in-Speech-using-CNN.git"
              demoLink={null}
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={require("../../Assets/PROJECT4.png")}
              isBlog={false}
              title="TabGANcf: Counterfactual GAN for Tabular Data"
              description="TabGANcf: Counterfactual GAN for Tabular Data. TabGANcf implements a Generative Adversarial Network designed to create high-quality, diverse counterfactual explanations for tabular datasets like Adult Income. The system uses a WGAN-GP architecture combined with classifier guidance to suggest data modifications that effectively flip model predictions while maintaining realistic feature distributions."
              ghLink="https://github.com/MuddasirKHuwaja998/TabGANcf-Counterfactual-GAN-for-Tabular-Data.git"
              demoLink={null}
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={require("../../Assets/project5.png")}
              isBlog={false}
              title="Breast Cancer Prediction System"
              description="Breast Cancer Prediction System. This project is a professional, production-ready web application that leverages Machine Learning (Logistic Regression) to predict whether a breast tumor is benign or malignant based on various cell sample features. The application is built with Python (Flask) for the backend, utilizes scikit-learn for modeling, and features a modern, dark-themed user interface styled with HTML and CSS. User Inputs: The user enters values for nine cell features, typically in the range 1–10."
              ghLink="https://github.com/MuddasirKHuwaja998/Breast-Cancer-Prediction-System.git"
              demoLink={null}
            />
          </Col>

          <Col md={4} className="project-card">
            <ProjectCard
              imgPath={require("../../Assets/project 6.png")}
              isBlog={false}
              title="Supervised Learning: Classification & Regression Models"
              description="Supervised Learning: Classification & Regression Models. This repository provides a professional implementation of eight core Supervised Learning models, covering both classification and regression tasks. It features Logistic Regression, K-NN, Naive Bayes, and SVM for categorical prediction, alongside Linear, Multiple, Polynomial, and Regularized (Ridge & Lasso) regression models for numerical forecasting."
              ghLink="https://github.com/MuddasirKHuwaja998/Machine-Learning-Supervised-Learning-Regression-.git"
              demoLink={null}
            />
          </Col>
        </Row>
      </Container>
    </Container>
  );
}

export default Projects;
