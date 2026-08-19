// Подсвечивает в содержании пункт, соответствующий секции под верхней кромкой
// экрана. Не через IntersectionObserver: у пунктов "assess_request" /
// "check_groundedness" нет своей <section> — это <h3> внутри секции
// "Инструменты" — а порог пересечения должен работать одинаково что для
// секций, что для этих вложенных подзаголовков.
(() => {
  const links = Array.prototype.slice.call(document.querySelectorAll('nav.toc a[href^="#"]'));
  const targets = links.map((link) => document.getElementById(link.getAttribute('href').slice(1)));

  const THRESHOLD = 96;
  let queued = false;

  const highlight = () => {
    queued = false;
    let active = -1;
    targets.forEach((target, i) => {
      if (target && target.getBoundingClientRect().top - THRESHOLD <= 0) {
        active = i;
      }
    });
    links.forEach((link, i) => {
      link.classList.toggle('active', i === active);
    });
  };

  const onScroll = () => {
    if (!queued) {
      queued = true;
      requestAnimationFrame(highlight);
    }
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  highlight();
})();

// Кнопка копирования в правом верхнем углу каждого блока с кодом. Текст
// снимается ДО того, как в <pre> добавляется сама кнопка — иначе разметка
// иконки попала бы в буфер обмена вместе с командой.
(() => {
  const COPY_ICON =
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
  const CHECK_ICON =
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>';

  const copyText = (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    // file:// — двойной клик по странице — не всегда secure context для
    // Clipboard API, поэтому запасной путь через скрытый textarea.
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    return Promise.resolve();
  };

  document.querySelectorAll('pre').forEach((pre) => {
    const code = pre.textContent;

    // Кнопка не может быть потомком <pre>: сам <pre> скроллится по
    // горизонтали (overflow-x: auto), и абсолютно спозиционированный внутри
    // него элемент уезжает вместе с содержимым. Оборачиваем pre в div, не
    // участвующий в этом скролле, и вешаем кнопку на него.
    const wrap = document.createElement('div');
    wrap.className = 'code-block';
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn';
    btn.setAttribute('aria-label', 'Скопировать');
    btn.innerHTML = COPY_ICON;
    btn.addEventListener('click', () => {
      copyText(code).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = CHECK_ICON;
        window.setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = COPY_ICON;
        }, 1400);
      });
    });
    wrap.appendChild(btn);
  });
})();
