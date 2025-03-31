import React, { useEffect, useState } from "react";
import { Bar, Pie, Scatter, Bubble } from "react-chartjs-2";
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from "chart.js";
import Papa from "papaparse";
import WordCloud from "react-d3-cloud";
import ChartDataLabels from "chartjs-plugin-datalabels";
import { PointElement } from 'chart.js';



// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement, ChartDataLabels, PointElement);

const Dashboard = () => {
  const [csvData, setCsvData] = useState([]);

  // Fetch CSV data from Flask backend
  useEffect(() => {
    const fetchCsvData = async () => {
      try {
        const response = await fetch("http://localhost:5001/csv");
        const reader = response.body.getReader();
        const result = await reader.read();
        const decoder = new TextDecoder("utf-8");
        const csvString = decoder.decode(result.value);
        const parsedData = Papa.parse(csvString, { header: true }).data;
        setCsvData(parsedData);
        console.log("Parsed CSV Data:", parsedData); // Debug: Log parsed data
      } catch (error) {
        console.error("Error fetching or parsing CSV:", error);
      }
    };

    fetchCsvData();
  }, []);

  // Process data for charts
  const processChartData = () => {
    const wordCounts = {};
    const emotionCounts = {};

    if (!csvData || csvData.length === 0) {
      console.warn("CSV data is empty or undefined.");
      return { wordCounts, emotionCounts };
    }

    csvData.forEach((row) => {
      // Process recognised_word
      const word = row.recognised_word?.trim() || "N/A";
      if (word !== "N/A") {
        wordCounts[word] = (wordCounts[word] || 0) + 1;
      }

      // Process recognised_emotion
      if (row.recognised_emotion) {
        try {
          const emotions = JSON.parse(row.recognised_emotion.replace(/'/g, '"'));
          console.log("Parsed Emotions:", emotions); // Debug: Log parsed emotions
          Object.entries(emotions).forEach(([emotion, count]) => {
            emotionCounts[emotion] = (emotionCounts[emotion] || 0) + count;
          });
        } catch (error) {
          console.error("Error parsing recognised_emotion:", error);
        }
      }
    });

    console.log("Word Counts:", wordCounts); // Debug: Log word counts
    console.log("Emotion Counts:", emotionCounts); // Debug: Log emotion counts

    return { wordCounts, emotionCounts };
  };

  const { wordCounts, emotionCounts } = processChartData();

  // Prepare data for the word cloud
  const wordCloudData = Object.entries(emotionCounts).map(([text, value]) => ({
    text,
    value,
  }));

  console.log("Word Cloud Data:", wordCloudData); // Debug: Log word cloud data

  const values = Object.values(wordCounts);
  const maxVal = Math.max(...values);
  const minVal = Math.min(...values);
  // Create a function to generate the gradient
  const gradient = (chart) => {
    const ctx = chart.ctx;  // Use chart's context
    const chartArea = chart.chartArea;

    // Check if chartArea is available
    if (!chartArea) return;

    // Create a linear gradient
    const gradientFill = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
    gradientFill.addColorStop(0, 'rgba(51, 27, 235, 0.6)'); // Start color
    gradientFill.addColorStop(1, 'rgba(136, 143, 223, 0.6)'); // End color

    return gradientFill;
  };

  // Bar chart data for recognised words
  const wordChartData = {
    labels: Object.keys(wordCounts),
    datasets: [
      {
        label: "Recognised Letters",
        data: Object.values(wordCounts),
        backgroundColor: (context) => gradient(context.chart), // Apply the gradient to each bar
      },
    ],
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "top",
        },
        tooltip: {
          callbacks: {
            label: (tooltipItem) => {
              const label = tooltipItem.label || "";
              const value = tooltipItem.raw;
              return `${label}: ${value}`;
            },
          },
        },
      },
    },
  };

  // Pie chart data for recognised emotions
  const emotionChartData = {
    labels: Object.keys(emotionCounts),
    datasets: [
      {
        label: "Recognised Emotions",
        data: Object.values(emotionCounts),
        backgroundColor: [
          "rgba(255, 99, 132, 0.6)",
          "rgba(54, 162, 235, 0.6)",
          "rgba(255, 206, 86, 0.6)",
          "rgba(75, 192, 192, 0.6)",
          "rgba(153, 102, 255, 0.6)",
        ],
        borderColor: [
          "rgba(255, 99, 132, 1)",
          "rgba(54, 162, 235, 1)",
          "rgba(255, 206, 86, 1)",
          "rgba(75, 192, 192, 1)",
          "rgba(153, 102, 255, 1)",
        ],
        borderWidth: 1,
      },
    ],
  };

  const pieChartOptions = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      datalabels: {
        color: "grey",
        formatter: (value, ctx) => {
          let label = ctx.chart.data.labels[ctx.dataIndex];
          return label + "\n" + value; // Show label + value
        },
        font: {
          size: 14,
        },
      },
    },
  };

  // Process the data for correlation (sign -> most frequent emotion)
  const correlationData = {};

  csvData.forEach((row) => {
    const sign = row.recognised_word?.trim();
    if (!sign) return;

    // Process the recognised_emotion data
    if (row.recognised_emotion) {
      try {
        const emotions = JSON.parse(row.recognised_emotion.replace(/'/g, '"'));
        const mostFrequentEmotion = Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b);

        // Assign the most frequent emotion to the sign
        if (correlationData[sign]) {
          correlationData[sign][mostFrequentEmotion[0]] = (correlationData[sign][mostFrequentEmotion[0]] || 0) + mostFrequentEmotion[1];
        } else {
          correlationData[sign] = { [mostFrequentEmotion[0]]: mostFrequentEmotion[1] };
        }
      } catch (error) {
        console.error("Error parsing recognised_emotion:", error);
      }
    }
  });


  /// Define emotion colors
  const emotionColors = {
    angry: "rgba(255, 99, 132, 1)", // Red
    happy: "rgba(75, 192, 192, 1)", // Green
    neutral: "rgba(54, 162, 235, 1)", // Blue
    surprise: "rgba(255, 206, 86, 1)", // Yellow
    sad: "rgba(153, 102, 255, 1)", // Purple
  };

  // Prepare correlation chart data
  const correlationChartData = {
    datasets: [
      {
        label: "Correlation Between Letter and Emotion",
        data: Object.entries(correlationData).map(([sign, emotions]) => {
          const mostFrequentEmotion = Object.entries(emotions).reduce((a, b) => a[1] > b[1] ? a : b);
          return {
            x: sign, // X-axis: Sign
            y: mostFrequentEmotion[1], // Y-axis: Frequency of most frequent emotion
            emotion: mostFrequentEmotion[0], // Emotion label
          };
        }),
        backgroundColor: "rgba(75, 192, 192, 1)",
        pointRadius: 5,
      },
    ],
  };

  // Scatter plot options
  const scatterPlotOptions = {
    responsive: true,
    layout: {
      padding: {
        left: 20,
        right: 20,
        top: 20,
        bottom: 20,
      },
    },
    scales: {
      x: {
        type: "category", // X-axis for categorical data (signs)
      },
      y: {
        type: "linear", // Y-axis for numerical values (frequency of emotion)
        min: 0,
        title: {
          display: true,
          text: "Frequency of Emotion",
        },
      },
    },
    plugins: {
      legend: {
        position: "top",
        labels: {
          generateLabels: (chart) => {
            // Get unique emotions from the dataset
            const emotions = [...new Set(chart.data.datasets[0].data.map((item) => item.emotion))];
            // Create legend labels with emotion-specific colors
            return emotions.map((emotion) => ({
              text: emotion, // Legend label text
              fillStyle: emotionColors[emotion] || "grey", // Legend color
              strokeStyle: emotionColors[emotion] || "grey", // Border color (optional)
              hidden: false, // Ensure the legend item is not hidden
            }));
          },
        },
      },
      tooltips: {
        callbacks: {
          label: (tooltipItem) => {
            const sign = tooltipItem.data.labels[tooltipItem.dataIndex];
            const frequency = tooltipItem.data.datasets[0].data[tooltipItem.dataIndex].y;
            const emotion = tooltipItem.data.datasets[0].data[tooltipItem.dataIndex].emotion;
            return `${sign}: ${emotion} (${frequency})`;
          },
        },
      },
      datalabels: {
        display: true,
        align: 'top',
        anchor: 'end',
        formatter: function (value) {
          return `${value.emotion}: ${value.y}`;
        },
        color: function (context) {
          // Get the emotion from the data point
          const emotion = context.dataset.data[context.dataIndex].emotion;
          // Return the corresponding color from the emotionColors mapping
          return emotionColors[emotion] || "grey"; // Default to grey if emotion is not found
        },
        font: {
          size: 12,
        },
        padding: {
          top: 4,
          left: 4,
        },
        offset: 10,
        rotation: -30,
      },
    },
  };

  // Process the data for frequent word-emotion pairs
  const wordEmotionPairs = {};

  // Process data for word-emotion pairs
  csvData.forEach((row) => {
    const word = row.recognised_word?.trim();
    if (!word) return;

    if (row.recognised_emotion) {
      try {
        const emotions = JSON.parse(row.recognised_emotion.replace(/'/g, '"'));

        // Iterate through emotions and create word-emotion pairs
        Object.entries(emotions).forEach(([emotion, count]) => {
          const pair = `${word} - ${emotion}`;
          wordEmotionPairs[pair] = (wordEmotionPairs[pair] || 0) + count;
        });
      } catch (error) {
        console.error("Error parsing recognised_emotion:", error);
      }
    }
  });

  // Sort the word-emotion pairs by frequency and select the top 10
  const topWordEmotionPairs = Object.entries(wordEmotionPairs)
    .sort((a, b) => b[1] - a[1]) // Sort by frequency in descending order
    .slice(0, 10); // Select top 10 pairs

  // Create gradient for the word-emotion chart
  const getWordEmotionGradient = (ctx, maxValue) => {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400); // Create vertical gradient
    gradient.addColorStop(0, "rgba(237, 27, 149, 0.6)"); // Min color (light green)
    gradient.addColorStop(1, "rgba(236, 140, 161, 0.6)"); // Max color (red)
    return gradient;
  };

  // Prepare word-emotion chart data
  const wordEmotionChartData = {
    labels: topWordEmotionPairs.map(([pair]) => pair),
    datasets: [
      {
        label: "Top 10 Word-Emotion Pairs",
        data: topWordEmotionPairs.map(([_, count]) => count),
        backgroundColor: (context) => {
          const maxValue = Math.max(...topWordEmotionPairs.map(([_, count]) => count));
          return getWordEmotionGradient(context.chart.ctx, maxValue); // Apply gradient based on max value
        },
      },
    ],
  };


  return (
    <div style={{ textAlign: "center", padding: "20px" }}>
      <h1>Dashboard</h1>
      <br></br>

      <div style={{ marginBottom: "40px" }}>
        <h2>No. of Recognised Letter From Sign Language</h2>
        <Bar
          data={wordChartData}
          options={{
            responsive: true,
            plugins: {
              legend: {
                position: "top",
              },
            },
          }}
        />
      </div>

      <div style={{ display: "grid", placeItems: "center" }}>
        <div style={{ marginBottom: "80px", width: "700px", height: "500px", justifyContent: "center", placeItems: "center",  }}>
          <h2>No. of Emotion Recognised From Emotion Recogniser</h2>
          <Pie data={emotionChartData} options={pieChartOptions} />
        </div>
      </div>


      <div style={{ marginBottom: "10px" }}>
        <h2>Recognised Emotion Word Cloud</h2>
        {wordCloudData.length > 0 ? (
          <WordCloud
            data={wordCloudData}
            width={600}
            height={250}
            font="Arial"
            fontSize={(word) => Math.log2(word.value) * 10}
            rotate={() => 0} // Keep words straight (no rotation)
          />
        ) : (
          <p>No emotion data available.</p>
        )}
      </div>
      <div style={{ textAlign: "center", }}>


        {/* Correlation with Emotion Chart */}
        <div style={{ marginBottom: "40px" }}>
          <h2>Correlation Between Recognized Letters and Emotions</h2>
          <Scatter
            data={correlationChartData}
            options={scatterPlotOptions}
          />
        </div>

        {/* Frequent Word-Emotion Pairs Chart */}
        <div style={{ marginBottom: "40px" }}>
          <h2>Top 10 Letter-Emotion Pairs</h2>
          <Bar
            data={wordEmotionChartData}
            options={{
              responsive: true,
              plugins: {
                legend: {
                  position: "top",
                },
                tooltip: {
                  callbacks: {
                    label: function (context) {
                      const pair = context.label; // Word-Emotion pair
                      const count = context.raw;
                      return `${pair}: ${count}`; // Show the word-emotion pair and count
                    },
                  },
                },
              },
            }}
          />

        </div>
      </div>
    </div>
  );
};

export default Dashboard;