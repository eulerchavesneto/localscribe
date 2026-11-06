// Gera o manifest.json de um curso da Desec Academy (player Panda Video).
//
// Como usar:
//   1. Faça login em https://academy.desecsecurity.com e abra qualquer aula do curso.
//   2. Abra o console do navegador (F12 -> Console), cole este arquivo inteiro e dê Enter.
//   3. Aguarde (~1 aula por segundo). Ao final o navegador baixa o manifest.json.
//      Não feche nem troque de página durante a coleta.
//
// O script só lê as páginas das aulas com a sua própria sessão, uma por vez e com pausa.
(async () => {
  const menu = document.querySelector('#menumodulos');
  if (!menu) return console.error('Abra uma página de aula (com o menu de módulos) antes de colar.');

  const clean = (s) => s.trim().replace(/\s+/g, ' ');
  const modules = [];
  let current = null;
  for (const el of menu.children) {
    if (el.classList.contains('ui-accordion-header')) {
      current = { module: clean(el.innerText), lessons: [] };
      modules.push(current);
    } else if (current) {
      el.querySelectorAll('a[href*="/aula/"]').forEach((a) =>
        current.lessons.push({ href: a.getAttribute('href'), title: clean(a.innerText) })
      );
    }
  }

  const total = modules.reduce((n, m) => n + m.lessons.length, 0);
  let done = 0;
  for (const m of modules) {
    for (const lesson of m.lessons) {
      try {
        const html = await (await fetch('/' + lesson.href, { credentials: 'same-origin' })).text();
        const video = html.match(/pandavideo\.com\.br\/embed\/\?v=([0-9a-f-]{36})/);
        const lib = html.match(/player-(vz-[a-z0-9]{8}-[a-z0-9]{3})/);
        lesson.video = video ? video[1] : null;
        lesson.lib = lib ? lib[1] : null;
      } catch (err) {
        lesson.error = String(err);
      }
      if (++done % 25 === 0) console.log(`${done}/${total}`);
      await new Promise((r) => setTimeout(r, 600));
    }
  }

  const blob = new Blob([JSON.stringify(modules, null, 1)], { type: 'application/json' });
  const link = Object.assign(document.createElement('a'), {
    href: URL.createObjectURL(blob),
    download: 'manifest.json',
  });
  link.click();
  console.log(`Pronto: ${done} aulas. manifest.json baixado.`);
})();
