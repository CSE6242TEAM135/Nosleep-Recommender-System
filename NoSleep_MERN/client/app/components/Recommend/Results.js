import React from "react";

class Results extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div>
        <h2> Returned Story</h2>
        <p>{this.props.results.body}</p>
        <h2> Score: {this.props.results.score}</h2>

      </div>
    )
  }
}
export default Results
