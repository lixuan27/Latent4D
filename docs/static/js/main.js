// Lazy-load gallery videos and autoplay only while on screen, to keep the page light.
(function () {
  const vids = document.querySelectorAll('video[data-src]');
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      const v = e.target;
      if (e.isIntersecting) {
        if (!v.src) {
          v.src = v.dataset.src;
          v.muted = true; v.loop = true; v.playsInline = true;
        }
        v.play().catch(() => {});
      } else {
        v.pause();
      }
    });
  }, { rootMargin: '200px', threshold: 0.1 });
  vids.forEach((v) => io.observe(v));
})();
