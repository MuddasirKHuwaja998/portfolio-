import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import myImg from "../../Assets/avatar.svg";
import pitchingSvg from "../../Assets/pitching_y6kw.svg";
import programmingSvg from "../../Assets/undraw_programming_j1zw.svg";
import Tilt from "react-parallax-tilt";

function Home2() {
  return (
    <>
      <Container fluid className="home-about-section" id="about">
        <Container>
          <Row>
            <Col md={7} className="home-about-description">
              <h1 style={{ fontSize: "2.6em" }}>
                LET ME <span className="purple"> INTRODUCE </span> MYSELF
              </h1>
              <p className="home-about-body">
                I am Muddsasir Khuwaja, a Machine Learning Engineer with a strong foundation in Python and AI model development. My journey includes pioneering work with Generative Adversarial Networks (GANs) and advanced deep learning techniques.
                <br />
                <br />
                Currently, I am actively engaged in the medical industry, training sophisticated machine learning models and leading innovative AI projects that drive real-world impact.
                <br />
                <br />
                My passion lies in transforming complex challenges into elegant solutions, and I am always eager to push the boundaries of artificial intelligence.
              </p>
            </Col>
            <Col md={3} className="myAvtar" style={{ marginLeft: '4em', display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
              <Tilt>
                <img src={programmingSvg} className="img-fluid" alt="avatar" style={{ maxHeight: '480px', width: '100%' }} />
              </Tilt>
            </Col>
            <Col md={3} className="myAvtar">
              
            </Col>
          </Row>
        </Container>
      </Container>
      <Container fluid className="home-about-section" id="certifications">
        <Container>
          <Row>
            <Col md={7} className="home-about-description">
              <h1 style={{ fontSize: "2.2em", marginTop: "2em" }}>
                <span className="purple">CERTIFICATIONS & ACHIEVEMENTS</span>
              </h1>
              <ul style={{ fontSize: "1.15em", marginTop: "1em", listStyle: "none", paddingLeft: 0 }}>
                <li style={{ marginBottom: "1em" }}>
                  <b>Supervised Machine Learning, Deep Learning-AI</b><br/>
                  <span style={{ color: '#888' }}>Stanford University (Online)</span>
                </li>
                <li style={{ marginBottom: "1em" }}>
                  <b>Public Financial Management</b><br/>
                  <span style={{ color: '#888' }}>International Monetary Fund (IMF)</span>
                </li>
                <li style={{ marginBottom: "1em" }}>
                  <b>Web Development using PHP and Laravel</b><br/>
                  <span style={{ color: '#888' }}>APTECH Computer Education</span>
                </li>
                <li style={{ marginBottom: "1em" }}>
                  <b>Developing AI Applications with Python and Flask</b><br/>
                  <span style={{ color: '#888' }}>IBM</span>
                </li>
                <li style={{ marginBottom: "1em" }}>
                  <b>Certified MS Excel Professional &amp; Power BI Tools</b><br/>
                  <span style={{ color: '#888' }}>Professional Certification</span>
                </li>
                <li style={{ marginBottom: "1em" }}>
                  <b>Health, Safety and Environment</b><br/>
                  <span style={{ color: '#888' }}>TUV Austria</span>
                </li>
                <li style={{ marginBottom: "1em" }}>
                  <b>ISO 26000:2010</b><br/>
                  <span style={{ color: '#888' }}>Global Standards</span>
                </li>
              </ul>
            </Col>
                <Col md={5} className="myAvtar" style={{ display: "flex", alignItems: "center", justifyContent: "flex-start" }}>
                  <img src={pitchingSvg} className="img-fluid" alt="certifications" style={{ maxHeight: "320px" }} />
                </Col>
          </Row>
        </Container>
      </Container>
    </>
  );
}
export default Home2;
