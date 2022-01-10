option = {
  title: {
    left: "center",
    text: "测试1错题直方图",
  },
  dataset: [
    {
      dimensions: ["number", "frequent"],
      source: [
        ["07", 1],
        ["10", 1],
        ["12", 1],
        ["18", 1],
        ["23", 1],
        ["39", 1],
        ["41", 1],
        ["55", 1],
        ["58", 1],
        ["61", 1],
        ["62", 1],
        ["63", 1],
        ["67", 1],
        ["68", 1],
        ["73", 1],
      ],
    },
    {
      transform: {
        type: "sort",
        config: { dimension: "frequent", order: "desc" },
      },
    },
  ],
  xAxis: {
    type: "category",
    axisLabel: { interval: 0, rotate: 30 },
  },
  yAxis: {},
  series: {
    type: "bar",
    encode: { x: "number", y: "frequent" },
    datasetIndex: 1,
  },
};

/////////////////////////////////////////////////////
