import React from "react";
import { Col, Row } from "react-bootstrap";
import { SiNextdotjs, SiSolidity } from "react-icons/si";
import { FaRust } from "react-icons/fa";
import AiTech from "../../Assets/TechIcons/ai-technology.png";
import BigData from "../../Assets/TechIcons/big-data.png";
import Coding from "../../Assets/TechIcons/coding.png";
import Computer from "../../Assets/TechIcons/computer.png";
import ComputerScience from "../../Assets/TechIcons/computer-science.png";
import DeepLearning from "../../Assets/TechIcons/deep-learning.png";
import DigitalAssistant from "../../Assets/TechIcons/digital-assistant.png";
import PythonIcon from "../../Assets/TechIcons/python.png";
import Server from "../../Assets/TechIcons/server.png";
import SupervisedLearning from "../../Assets/TechIcons/supervised-learning.png";

function Techstack() {
  return (
    <>
      <Row style={{ justifyContent: "center", paddingBottom: "10px" }}>
        <Col xs={6} md={2} className="tech-icons">
          <img src={SupervisedLearning} alt="Machine Learning" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Machine Learning (Supervised & Unsupervised)</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={DeepLearning} alt="Deep Learning" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Deep Learning (CNNs, GANs, AlexNet, DICE, Random Forest)</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={BigData} alt="Big Data" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Big Data Analytics</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={PythonIcon} alt="Python" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Python (advanced)</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={Coding} alt="Data Science" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Data Science & Data Analysis</div>
        </Col>
      </Row>
      <Row style={{ justifyContent: "center", paddingBottom: "10px" }}>
        <Col xs={6} md={2} className="tech-icons">
          <img src={Computer} alt="Computer Vision" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Computer Vision & NLP</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={AiTech} alt="AI Model" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">AI Model Development & Deployment</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={ComputerScience} alt="Web Development" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Web Development (PHP, Laravel, Flask)</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={Server} alt="Cloud" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Cloud & Vertex AI</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={DigitalAssistant} alt="Virtual Assistant" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Virtual Assistants & Automation</div>
        </Col>
      </Row>
    </>
  );
}

export default Techstack;
