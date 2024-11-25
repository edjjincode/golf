<template>
  <div class="golf-analysis">
    <div class="video-container">
      <!-- 비디오 업로드 -->
      <input type="file" @change="onVideoUpload" accept="video/*" />

      <!-- 비디오 미리보기 -->
      <video v-if="uploadedVideo" ref="video" :src="uploadedVideo" controls></video>

      <!-- Analyze Video 버튼 -->
      <button @click="analyzeVideo" :disabled="!uploadedVideo || isAnalyzing">
        {{ isAnalyzing ? "Analyzing..." : "Analyze Video" }}
      </button>

      <!-- 분석 결과 출력 -->
      <div v-if="analysisResult" class="analysis-result">
        <h2>Analysis Results</h2>
        <pre>{{ analysisResult }}</pre>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "GolfAnalysis",
  data() {
    return {
      uploadedVideo: null, // 업로드된 비디오 URL
      isAnalyzing: false, // 분석 진행 여부
      websocket: null, // WebSocket 연결 객체
      analysisResult: null, // 분석 결과 텍스트
    };
  },
  methods: {
    onVideoUpload(event) {
      const file = event.target.files[0];
      if (file) {
        if (file.type.startsWith("video/")) {
          this.uploadedVideo = URL.createObjectURL(file); // 미리보기 URL 생성
        } else {
          alert("Please upload a valid video file."); // 비디오 파일이 아닌 경우 경고
        }
      }
    },
    analyzeVideo() {
      if (!this.uploadedVideo) {
        alert("Please upload a video first.");
        return;
      }

      this.isAnalyzing = true;
      this.analysisResult = null;

      const video = this.$refs.video;
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      this.websocket = new WebSocket("ws://localhost:8000/ws");

      this.websocket.onopen = () => {
        console.log("WebSocket connection established");
        video.play();
      };

      this.websocket.onmessage = (event) => {
        const data = event.data;

        if (data.startsWith("Error:")) {
          console.error(data);
          alert(data); // 에러 메시지를 사용자에게 표시
        } else if (data.includes("\n")) {
          // 분석 결과 텍스트 수신
          this.analysisResult = data;
          this.isAnalyzing = false;
        } else {
          console.log("Received intermediate data:", data);
        }
      };

      this.websocket.onclose = () => {
        console.log("WebSocket connection closed");
        this.isAnalyzing = false;
      };

      video.onplay = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const sendFrame = () => {
          if (video.paused || video.ended) {
            if (this.websocket.readyState === WebSocket.OPEN) {
              this.websocket.send("end");
            }
            return;
          }

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const frameData = canvas.toDataURL("image/jpeg");
          if (this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(frameData);
          }

          requestAnimationFrame(sendFrame);
        };

        sendFrame();
      };
    },
  },
};
</script>

<style scoped>
.golf-analysis {
  text-align: center;
}
.video-container {
  margin: 20px auto;
  max-width: 800px;
}
video {
  width: 100%;
  height: auto;
}
.analysis-result {
  margin-top: 20px;
  padding: 10px;
  background-color: #f9f9f9;
  border: 1px solid #ddd;
  white-space: pre-wrap; /* 텍스트 줄바꿈 지원 */
}
</style>
