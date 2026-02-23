import React from "react";
import Card from "react-bootstrap/Card";
import { ImPointRight } from "react-icons/im";

function AboutCard() {
  return (
    <Card className="quote-card-view">
      <Card.Body>
        <blockquote className="blockquote mb-0">
          <p style={{ textAlign: "justify" }}>
            Hello! I’m <span className="purple">Muddasir Khuwaja</span>.<br/>
            I hold an <span className="purple">MSc in Applied Computer Science (ML &amp; Big Data)</span> and a <span className="purple">BE in Computer System Engineering</span>.<br/>
            <br/>
            My journey is dedicated to advancing <span className="purple">machine learning</span> every day, with a special passion for <span className="purple">unsupervised learning</span> and deep learning research.<br/>
            <br/>
            I work extensively with models such as <span className="purple">CNNs</span> (including <span className="purple">AlexNet</span>), <span className="purple">Random Forest</span>, <span className="purple">DICE</span>, and <span className="purple">GANs</span>, applying them to real-world problems and innovative projects.<br/>
            <br/>
            My focus is on pushing the boundaries of AI, exploring new architectures, and making a meaningful impact through intelligent systems.
          </p>

          <p style={{ color: "rgb(155 126 172)", marginTop: "2em" }}>
            "Every epoch is a new opportunity to learn, innovate, and redefine what’s possible with AI."
          </p>
          <footer className="blockquote-footer">Muddasir Khuwaja</footer>
        </blockquote>
      </Card.Body>
    </Card>
  );
}

export default AboutCard;
