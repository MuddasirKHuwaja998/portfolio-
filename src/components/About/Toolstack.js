import React from "react";
import { Col, Row } from "react-bootstrap";
import VSCode from "../../Assets/TechIcons/visual-studio-code-svgrepo-com.svg";
import GitHub from "../../Assets/TechIcons/github.png";
import PyCharm from "../../Assets/TechIcons/intellij_pycharm_alt_macos_bigsur_icon_190054.png";
import Jupyter from "../../Assets/TechIcons/jupyter_app_icon_161280.svg";
import Kaggle from "../../Assets/TechIcons/kaggle-svgrepo-com.svg";
import Colab from "../../Assets/TechIcons/icons8-google-colab-48.png";

function Toolstack() {
  return (
    <>
      <Row style={{ justifyContent: "center", paddingBottom: "10px" }}>
        <Col xs={6} md={2} className="tech-icons">
          <img src={VSCode} alt="VS Code" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">VS Code</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={GitHub} alt="GitHub" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">GitHub</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={PyCharm} alt="PyCharm" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">PyCharm</div>
        </Col>
      </Row>
      <Row style={{ justifyContent: "center", paddingBottom: "10px" }}>
        <Col xs={6} md={2} className="tech-icons">
          <img src={Jupyter} alt="Jupyter" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Jupyter Notebook</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={Kaggle} alt="Kaggle" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Kaggle</div>
        </Col>
        <Col xs={6} md={2} className="tech-icons">
          <img src={Colab} alt="Google Colab" style={{height: '48px', width: '48px'}} />
          <div className="tech-icons-text">Google Colab</div>
        </Col>
      </Row>
    </>
  );
}

export default Toolstack;
