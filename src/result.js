option = {
  title: {
    left: "center",
    text: "测试1错题直方图",
  },
  dataset: [
    {
      dimensions: ["number", "frequent"],
      source: [
        ["07", 2],
        ["08", 2],
        ["10", 2],
        ["12", 1],
        ["17", 1],
        ["18", 1],
        ["19", 1],
        ["22", 1],
        ["23", 1],
        ["27", 1],
        ["29", 1],
        ["30", 1],
        ["36", 1],
        ["39", 1],
        ["41", 1],
        ["44", 1],
        ["46", 1],
        ["51", 1],
        ["52", 1],
        ["55", 2],
        ["58", 1],
        ["59", 1],
        ["61", 2],
        ["62", 1],
        ["63", 2],
        ["67", 1],
        ["68", 2],
        ["73", 2],
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

option = {
  title: {
    left: "center",
    text: "测试2错题直方图",
  },
  dataset: [
    {
      dimensions: ["number", "frequent"],
      source: [
        ["01", 1],
        ["06", 2],
        ["07", 1],
        ["10", 1],
        ["12", 1],
        ["13", 1],
        ["16", 2],
        ["19", 1],
        ["21", 1],
        ["22", 1],
        ["25", 1],
        ["26", 2],
        ["28", 1],
        ["29", 1],
        ["42", 1],
        ["43", 1],
        ["49", 2],
        ["55", 1],
        ["64", 2],
        ["66", 1],
        ["68", 1],
        ["71", 2],
        ["72", 1],
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
