import React from "react";
function Pre(props) {
  return (
    <div id={props.load ? "preloader" : "preloader-none"} style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh" }}>
      <img src={require("../Assets/pre.gif")} alt="preloader-logo" style={{ width: "14em", height: "8em" }} />
    </div>
  );
}

export default Pre;
