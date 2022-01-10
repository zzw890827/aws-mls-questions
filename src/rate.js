option = {
  title: {
    left: "center",
    text: "测试1正确率推移",
  },
  tooltip: {
    trigger: "axis",
  },
  xAxis: {
    type: "category",
    data: ["第一回"],
  },
  yAxis: {
    type: "value",
    scale: true,
    axisLabel: {
      formatter: "{value}%",
    },
  },
  series: [
    {
      data: [80.3],
      type: "line",
      markLine: {
        symbol: ["none", "none"],
        label: { show: true },
        data: [{ yAxis: 75 }],
      },
      markPoint: {
        data: [
          { type: "max", name: "最大值" },
          { type: "min", name: "最小值" },
        ],
      },
    },
  ],
};
