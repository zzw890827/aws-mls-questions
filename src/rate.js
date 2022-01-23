option = {
  title: {
    left: 'center',
    text: '测试1正确率推移',
  },
  tooltip: {
    trigger: 'axis',
  },
  xAxis: {
    type: 'category',
    data: ['第一回', '第二回', '第三回', '第四回'],
  },
  yAxis: {
    type: 'value',
    scale: true,
    axisLabel: {
      formatter: '{value}%',
    },
  },
  series: [
    {
      data: [80.3, 73.7, 80.3, 90.8],
      type: 'line',
      markLine: {
        symbol: ['none', 'none'],
        label: { show: true },
        data: [{ yAxis: 75 }],
      },
      markPoint: {
        data: [
          { type: 'max', name: '最大值' },
          { type: 'min', name: '最小值' },
        ],
      },
    },
  ],
};

//

option = {
  title: {
    left: 'center',
    text: '测试2正确率推移',
  },
  tooltip: {
    trigger: 'axis',
  },
  xAxis: {
    type: 'category',
    data: ['第一回', '第二回', '第三回', '第四回'],
  },
  yAxis: {
    type: 'value',
    scale: true,
    axisLabel: {
      formatter: '{value}%',
    },
  },
  series: [
    {
      data: [76.3, 85.5, 89.5, 93.4],
      type: 'line',
      markLine: {
        symbol: ['none', 'none'],
        label: { show: true },
        data: [{ yAxis: 75 }],
      },
      markPoint: {
        data: [
          { type: 'max', name: '最大值' },
          { type: 'min', name: '最小值' },
        ],
      },
    },
  ],
};

//

option = {
  title: {
    left: 'center',
    text: '测试3正确率推移',
  },
  tooltip: {
    trigger: 'axis',
  },
  xAxis: {
    type: 'category',
    data: ['第一回', '第二回', '第三回',],
  },
  yAxis: {
    type: 'value',
    scale: true,
    axisLabel: {
      formatter: '{value}%',
    },
  },
  series: [
    {
      data: [69.7, 90.8, 94.7],
      type: 'line',
      markLine: {
        symbol: ['none', 'none'],
        label: { show: true },
        data: [{ yAxis: 75 }],
      },
      markPoint: {
        data: [
          { type: 'max', name: '最大值' },
          { type: 'min', name: '最小值' },
        ],
      },
    },
  ],
};
