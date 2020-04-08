const AWS = require('aws-sdk');
const config = require('../../../config/config.js');
const isDev = process.env.NODE_ENV !== 'production';

module.exports = (app) => {
  //get all processed stories
  app.get('/api/stories', (req, res, next) => {
    if (isDev) {
      console.log('isDev');
      AWS.config.update(config.aws_local_config);
    } else {
      console.log('isProd');
      AWS.config.update(config.aws_remote_config);
    }
    const docClient = new AWS.DynamoDB.DocumentClient();
    const params = {
      TableName: "StoriesNoSleep"
    };
    docClient.scan(params, function(err, data) {
      if (err) {
        res.send({
          success: false,
          message: 'Error: Server error'
        });
      } else {
        const {Items} = data;
        res.send({
          success: true,
          message: 'Loaded Stories',
          stories: Items
        });
      }
    });
  });
  //get Userstory by ID
  app.get('/api/story', (req, res, next) => {
    if (isDev) {
      AWS.config.update(config.aws_local_config);
    } else {
      AWS.config.update(config.aws_remote_config);
    }
    const storyId = req.query.id;
    const docClient = new AWS.DynamoDB.DocumentClient();
    const params = {
      TableName: 'UserStories',
      KeyConditionExpression: 'story_id = :i',
      ExpressionAttributeValues: {
        ':i': storyId
      }
    };
    docClient.query(params, function(err, data) {
      if (err) {
        res.send({
          success: false,
          message: 'Error: Server error'
        });
      } else {
        const { Items } = data;
        res.send({
          success: true,
          message: 'Loaded The User Story by ID',
          story: Items
        });
      }
    });
  });
  //get Story and Score
  app.get('/api/storyScore', (req, res, next) => {
    if (isDev) {
      AWS.config.update(config.aws_local_config);
    } else {
      AWS.config.update(config.aws_remote_config);
    }
    const storyId = req.query.id;
    const docClient = new AWS.DynamoDB.DocumentClient();
    const params = {
      TableName: 'StoriesNoSleep',
      ProjectionExpression: 'score',
      KeyConditionExpression: 'id = :i',
      ExpressionAttributeValues: {
        ':i': storyId
      }
    };
    docClient.query(params, function(err, data) {
      if (err) {
        res.send({
          success: false,
          message: 'Error: Server error'
        });
      } else if (data.Count == 0) {
        res.send({
          success: true,
          message: 'Story Not Processed'
        })

      } else {
        const { Items } = data;
        res.send({
          success: true,
          message: 'Loaded The Processed Story by ID',
          story: Items
        });
      }
    });
  });
};
