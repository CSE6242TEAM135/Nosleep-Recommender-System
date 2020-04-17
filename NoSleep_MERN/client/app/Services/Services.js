module.exports = {
  getStoryID: url => {
  //  https://www.reddit.com/r/nosleep/comments/fvdz0f/my_brother_kept_a_dream_diary_i_wish_i_hadnt_read/
    var storyID = url.match(/(?<=\/comments\/)([^\/]*)/)[0]
    console.log(storyID)
    return storyID
  }
}
