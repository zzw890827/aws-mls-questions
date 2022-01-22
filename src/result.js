option = {
  title: {
    left: 'center',
    text: '测试1错题直方图',
  },
  dataset: [
    {
      dimensions: ['number', 'frequent'],
      source: [
        ['07', 3],
        ['08', 2],
        ['10', 2],
        ['12', 1],
        ['17', 2],
        ['18', 1],
        ['19', 2],
        ['22', 1],
        ['23', 1],
        ['24', 1],
        ['27', 1],
        ['29', 1],
        ['30', 2],
        ['36', 1],
        ['38', 1],
        ['39', 2],
        ['41', 1],
        ['44', 1],
        ['45', 1],
        ['46', 2],
        ['51', 2],
        ['52', 2],
        ['55', 2],
        ['58', 1],
        ['59', 2],
        ['61', 2],
        ['62', 1],
        ['63', 3],
        ['67', 1],
        ['68', 2],
        ['70', 1],
        ['73', 3],
      ],
    },
    {
      transform: {
        type: 'sort',
        config: { dimension: 'frequent', order: 'desc' },
      },
    },
  ],
  xAxis: {
    type: 'category',
    axisLabel: { interval: 0, rotate: 30 },
  },
  yAxis: {},
  series: {
    type: 'bar',
    encode: { x: 'number', y: 'frequent' },
    datasetIndex: 1,
  },
};

/////////////////////////////////////////////////////

option = {
  title: {
    left: 'center',
    text: '测试2错题直方图',
  },
  dataset: [
    {
      dimensions: ['number', 'frequent'],
      source: [
        ['01', 1],
        ['06', 2],
        ['07', 1],
        ['10', 2],
        ['12', 1],
        ['13', 1],
        ['16', 2],
        ['19', 1],
        ['21', 2],
        ['22', 2],
        ['25', 1],
        ['26', 3],
        ['28', 2],
        ['29', 1],
        ['42', 2],
        ['43', 1],
        ['46', 1],
        ['49', 2],
        ['55', 1],
        ['64', 2],
        ['66', 1],
        ['68', 1],
        ['71', 3],
        ['72', 1],
      ],
    },
    {
      transform: {
        type: 'sort',
        config: { dimension: 'frequent', order: 'desc' },
      },
    },
  ],
  xAxis: {
    type: 'category',
    axisLabel: { interval: 0, rotate: 30 },
  },
  yAxis: {},
  series: {
    type: 'bar',
    encode: { x: 'number', y: 'frequent' },
    datasetIndex: 1,
  },
};

/////////////////////////////////////////////////////

option = {
  title: {
    left: 'center',
    text: '测试3错题直方图',
  },
  dataset: [
    {
      dimensions: ['number', 'frequent'],
      source: [
        ['03', 1],
        ['05', 1],
        ['07', 1],
        ['08', 1],
        ['10', 2],
        ['11', 1],
        ['13', 1],
        ['22', 2],
        ['26', 2],
        ['28', 1],
        ['29', 1],
        ['30', 2],
        ['31', 1],
        ['36', 1],
        ['43', 2],
        ['47', 2],
        ['55', 1],
        ['58', 1],
        ['59', 1],
        ['61', 1],
        ['64', 1],
        ['71', 1],
        ['72', 1],
        ['73', 1],
        ['75', 2],
      ],
    },
    {
      transform: {
        type: 'sort',
        config: { dimension: 'frequent', order: 'desc' },
      },
    },
  ],
  xAxis: {
    type: 'category',
    axisLabel: { interval: 0, rotate: 30 },
  },
  yAxis: {},
  series: {
    type: 'bar',
    encode: { x: 'number', y: 'frequent' },
    datasetIndex: 1,
  },
};
