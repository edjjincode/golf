<template>
  <div id="app">
    <header>
      <h1>Golf Pose Analysis</h1>
    </header>
    <main>
      <!-- Golf Analysis Component -->
      <GolfAnalysis />
    </main>
    <footer>
      <p>© 2024 Golf Analysis Team</p>
    </footer>
  </div>
</template>

<script>
import GolfAnalysis from "./components/GolfAnalysis.vue"; // GolfAnalysis 컴포넌트 import

export default {
  name: "App",
  components: {
    GolfAnalysis,
  },
};
</script>

<style>
/* 전역 스타일 */
#app {
  font-family: Arial, sans-serif;
  text-align: center;
  color: #333;
  margin: 0;
  padding: 0;
}

header {
  background-color: #42b983;
  color: white;
  padding: 20px 0;
}

main {
  padding: 20px;
}

footer {
  background-color: #f0f0f0;
  color: #666;
  padding: 10px 0;
  margin-top: 20px;
}
</style>