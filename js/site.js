(function () {
  var hero = document.querySelector('.hero');
  var finePointer = window.matchMedia('(pointer: fine)').matches;

  if (hero && finePointer) {
    hero.addEventListener('pointermove', function (event) {
      var bounds = hero.getBoundingClientRect();
      hero.style.setProperty('--pointer-x', (event.clientX - bounds.left) + 'px');
      hero.style.setProperty('--pointer-y', (event.clientY - bounds.top) + 'px');
    });

    hero.addEventListener('pointerleave', function () {
      hero.style.removeProperty('--pointer-x');
      hero.style.removeProperty('--pointer-y');
    });
  }

  var navToggle = document.querySelector('.nav-toggle');
  var navigation = document.getElementById('site-navigation');

  if (navToggle && navigation) {
    navToggle.addEventListener('click', function () {
      var open = navToggle.getAttribute('aria-expanded') === 'true';
      navToggle.setAttribute('aria-expanded', String(!open));
      navToggle.setAttribute('aria-label', open ? 'Open navigation' : 'Close navigation');
      navigation.classList.toggle('is-open', !open);
      document.body.classList.toggle('nav-is-open', !open);
    });

    navigation.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        navToggle.setAttribute('aria-expanded', 'false');
        navToggle.setAttribute('aria-label', 'Open navigation');
        navigation.classList.remove('is-open');
        document.body.classList.remove('nav-is-open');
      });
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && navToggle.getAttribute('aria-expanded') === 'true') {
        navToggle.click();
        navToggle.focus();
      }
    });

    window.addEventListener('resize', function () {
      if (window.innerWidth > 672 && navToggle.getAttribute('aria-expanded') === 'true') {
        navToggle.setAttribute('aria-expanded', 'false');
        navToggle.setAttribute('aria-label', 'Open navigation');
        navigation.classList.remove('is-open');
        document.body.classList.remove('nav-is-open');
      }
    });
  }

  var trajectory = document.querySelector('.trajectory-map');
  var threadCanvas = document.querySelector('.trajectory-canvas');

  if (trajectory && threadCanvas) {
    var threadContext = threadCanvas.getContext('2d');
    var activeMilestone = '';
    var threadWidth = 0;
    var threadHeight = 0;
    var threadRatio = 1;
    var reduceThreadMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    var threadPointer = {
      x: 0,
      y: 0,
      targetX: 0,
      targetY: 0,
      strength: 0,
      targetStrength: 0
    };

    function clamp(value, minimum, maximum) {
      return Math.max(minimum, Math.min(maximum, value));
    }

    function resizeThread() {
      var bounds = trajectory.getBoundingClientRect();
      threadWidth = bounds.width;
      threadHeight = bounds.height;
      threadRatio = Math.min(window.devicePixelRatio || 1, 2);
      threadCanvas.width = Math.round(threadWidth * threadRatio);
      threadCanvas.height = Math.round(threadHeight * threadRatio);
      threadContext.setTransform(threadRatio, 0, 0, threadRatio, 0, 0);
      if (reduceThreadMotion) drawThreads(0);
    }

    function milestonePoints(selector, baseX, time, compact) {
      var mapBounds = trajectory.getBoundingClientRect();
      return Array.prototype.map.call(trajectory.querySelectorAll(selector), function (card, index) {
        var cardBounds = card.getBoundingClientRect();
        var driftAmount = compact ? 2.5 : 8;
        var drift = reduceThreadMotion ? 0 : Math.sin(time / 1500 + index * 1.7) * driftAmount;
        var point = {
          card: card,
          key: card.getAttribute('data-milestone'),
          x: baseX + drift,
          y: cardBounds.top - mapBounds.top + cardBounds.height / 2,
          anchorX: selector.indexOf('work') > -1 ?
            cardBounds.left - mapBounds.left : cardBounds.right - mapBounds.left
        };

        if (threadPointer.strength > .001) {
          var distanceY = point.y - threadPointer.y;
          var influence = Math.exp(-(distanceY * distanceY) / 64800) * threadPointer.strength;
          point.x += clamp((threadPointer.x - baseX) * .16, -56, 56) * influence;
        }

        return point;
      });
    }

    function strokeThread(points, color) {
      if (!points.length) return;
      threadContext.beginPath();
      threadContext.moveTo(points[0].x, points[0].y);
      for (var index = 1; index < points.length; index += 1) {
        var previous = points[index - 1];
        var current = points[index];
        var middleY = (previous.y + current.y) / 2;
        threadContext.bezierCurveTo(previous.x, middleY, current.x, middleY, current.x, current.y);
      }
      threadContext.strokeStyle = color;
      threadContext.lineWidth = 2;
      threadContext.shadowColor = color;
      threadContext.shadowBlur = 10;
      threadContext.stroke();
      threadContext.shadowBlur = 0;
    }

    function drawTrack(points, color, startX) {
      var pathPoints = [{ x: startX, y: 0 }].concat(points.map(function (point) {
        return { x: point.x, y: point.y };
      })).concat([{ x: startX, y: threadHeight }]);

      if (threadPointer.strength > .001 && threadPointer.y > 0 && threadPointer.y < threadHeight) {
        pathPoints.push({
          x: startX + clamp((threadPointer.x - startX) * .14, -68, 68) * threadPointer.strength,
          y: threadPointer.y
        });
        pathPoints.sort(function (first, second) { return first.y - second.y; });
      }

      strokeThread(pathPoints, color);

      points.forEach(function (point) {
        var active = point.key === activeMilestone;
        threadContext.beginPath();
        threadContext.moveTo(point.anchorX, point.y);
        threadContext.bezierCurveTo(
          (point.anchorX + point.x) / 2, point.y,
          (point.anchorX + point.x) / 2, point.y,
          point.x, point.y
        );
        threadContext.strokeStyle = active ? color : color.replace('1)', '.22)');
        threadContext.lineWidth = active ? 1.5 : 1;
        threadContext.setLineDash(active ? [] : [4, 5]);
        threadContext.stroke();
        threadContext.setLineDash([]);

        if (active) {
          threadContext.beginPath();
          threadContext.arc(point.x, point.y, 13, 0, Math.PI * 2);
          threadContext.fillStyle = color.replace('1)', '.1)');
          threadContext.fill();
        }

        threadContext.beginPath();
        threadContext.arc(point.x, point.y, active ? 6 : 4.5, 0, Math.PI * 2);
        threadContext.fillStyle = '#080b12';
        threadContext.strokeStyle = color;
        threadContext.lineWidth = active ? 3 : 2;
        threadContext.shadowColor = active ? color : 'transparent';
        threadContext.shadowBlur = active ? 16 : 0;
        threadContext.fill();
        threadContext.stroke();
        threadContext.shadowBlur = 0;
      });
    }

    function drawThreads(time) {
      threadPointer.x += (threadPointer.targetX - threadPointer.x) * .09;
      threadPointer.y += (threadPointer.targetY - threadPointer.y) * .09;
      threadPointer.strength += (threadPointer.targetStrength - threadPointer.strength) * .075;
      threadContext.clearRect(0, 0, threadWidth, threadHeight);
      var compact = window.innerWidth <= 928;
      var workX = compact ? 12 : 18;
      var studyX = compact ? threadWidth - 12 : threadWidth - 18;
      var workPoints = milestonePoints('.trajectory-work', workX, time, compact);
      var studyPoints = milestonePoints('.trajectory-study', studyX, time + 700, compact);
      drawTrack(workPoints, 'rgba(255, 115, 92, 1)', workX);
      drawTrack(studyPoints, 'rgba(154, 167, 255, 1)', studyX);
      if (!reduceThreadMotion) window.requestAnimationFrame(drawThreads);
    }

    document.documentElement.classList.add('has-trajectory');
    if (finePointer && !reduceThreadMotion) {
      trajectory.addEventListener('pointermove', function (event) {
        var bounds = trajectory.getBoundingClientRect();
        threadPointer.targetX = event.clientX - bounds.left;
        threadPointer.targetY = event.clientY - bounds.top;
        if (threadPointer.strength === 0) {
          threadPointer.x = threadPointer.targetX;
          threadPointer.y = threadPointer.targetY;
        }
        threadPointer.targetStrength = 1;
      });

      trajectory.addEventListener('pointerleave', function () {
        threadPointer.targetStrength = 0;
      });
    }

    trajectory.querySelectorAll('[data-milestone]').forEach(function (card) {
      function activate() { activeMilestone = card.getAttribute('data-milestone'); if (reduceThreadMotion) drawThreads(0); }
      function deactivate() { activeMilestone = ''; if (reduceThreadMotion) drawThreads(0); }
      card.addEventListener('pointerenter', activate);
      card.addEventListener('pointerleave', deactivate);
      card.addEventListener('focus', activate);
      card.addEventListener('blur', deactivate);
    });

    window.addEventListener('resize', resizeThread);
    resizeThread();
    if (!reduceThreadMotion) window.requestAnimationFrame(drawThreads);
  }

  var editorialPage = document.querySelector('.publications-page, .talks-page');
  var reducePageMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  if (editorialPage && finePointer && !reducePageMotion) {
    window.addEventListener('pointermove', function (event) {
      editorialPage.style.setProperty('--page-x', event.clientX + 'px');
      editorialPage.style.setProperty('--page-y', event.clientY + 'px');
    });

    editorialPage.querySelectorAll('.post-content > p:not(.page-deck)').forEach(function (item) {
      item.addEventListener('pointermove', function (event) {
        var bounds = item.getBoundingClientRect();
        var localX = event.clientX - bounds.left;
        var localY = event.clientY - bounds.top;
        item.style.setProperty('--item-x', localX + 'px');
        item.style.setProperty('--item-y', localY + 'px');

        if (editorialPage.classList.contains('talks-page')) {
          item.style.setProperty('--tilt-x', ((bounds.height / 2 - localY) / bounds.height * 2.2) + 'deg');
          item.style.setProperty('--tilt-y', ((localX - bounds.width / 2) / bounds.width * 2.2) + 'deg');
        }
      });

      item.addEventListener('pointerleave', function () {
        item.style.removeProperty('--tilt-x');
        item.style.removeProperty('--tilt-y');
      });
    });
  }

  if ('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    document.documentElement.classList.add('has-reveal');
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: .08 });

    document.querySelectorAll('.reveal-section').forEach(function (section) {
      observer.observe(section);
    });
  }
}());
