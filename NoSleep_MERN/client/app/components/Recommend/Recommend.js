import React from "react";
import Services from "../../Services/Services"
import axios from 'axios'
import Results from "./Results"

class Recommend extends React.Component {
  constructor(props) {
    super(props);
    this.state = {value: '',
                  storyId: "",
                  title: "",
                  body: "",
                  author: "",
                  score:"",
                  isSubmitted: false};


    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    this.setState({value: event.target.value});
  }

  handleSubmit(event) {
    // alert('A story was submitted: ' + this.state.value);
    const storyID = Services.getStoryID(this.state.value);
    axios.get(`/api/story?id=${storyID}`)
      .then(res => {
        const item = res.data.story[0]
        if (item != null) {
          this.setState({ storyId: item.story_id})
          this.setState({ title: item.title})
          this.setState({ body: item.body})
          this.setState({ author: item.author})
        } else {
          this.setState({ body: "Story not Found"})
        }

        this.setState({isSubmitted: true})
      })
    axios.get(`/api/storyScore?id=${storyID}`)
      .then(res => {
        const item2 = res.data
        if (item2.message === "Story Not Processed") {
          this.setState({ score: item2.message})
        } else {
          this.setState({ score: item2.story[0].score})
        }
        window.results = res.data

      })
    event.preventDefault();
  }

  render() {
    return (
      <div>
      <form onSubmit={this.handleSubmit}>
        <label>
          No Sleep Reddit Story Link:
          <input type="text" value={this.state.value} onChange={this.handleChange} />
        </label>
        <input type="submit" value="Recommend" />
      </form>
        {this.state.isSubmitted && <Results results={this.state}/>}
      </div>
    )
  }
}

export default Recommend
