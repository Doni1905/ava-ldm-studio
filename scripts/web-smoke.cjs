const { chromium } = require("playwright");
const assert = require("assert");
(async () => {
  let count = 0;
  const b = await chromium.launch({ args: ["--no-sandbox"] });
  for (const width of [1280, 390, 320]) {
    const p = await b.newPage({ viewport: { width, height: 844 }, userAgent: "AVA-LDM-Android" });
    const errors = [];
    p.on("pageerror", (e) => errors.push(e.message));
    await p.goto(process.env.STUDIO_URL || "http://localhost:5174/assets/index.html");
    await p.getByRole("button", { name: "Run LDM Analysis" }).waitFor();
    for (const text of ["Do not call mother", "அம்மாவுக்கு அழை", "Call mother"]) {
      await p.locator("textarea").fill(text);
      await p.getByRole("button", { name: "Run LDM Analysis" }).click();
      await p.waitForTimeout(150);
      const x = JSON.parse(await p.locator("pre").innerText());
      if (text.startsWith("Do not")) assert.equal(x.intent, "UNKNOWN");
      if (text === "அம்மாவுக்கு அழை") assert.equal(x.intent, "MAKE_CALL");
      count++;
    }
    for (const route of ["dataset", "evaluation", "pipeline"]) {
      await p.getByRole("link", { name: route, exact: false }).last().click();
      await p.waitForTimeout(200);
      if (route === "evaluation") {
        await p.getByRole("button", { name: "Re-evaluate Benchmark" }).click();
        await p.getByText("30 samples evaluated", { exact: false }).waitFor();
      }
      assert(!(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth)));
      await p.screenshot({ path: "/tmp/final-" + width + "-" + route + ".png", fullPage: true });
      count++;
    }
    await p.getByRole("link", { name: "Playground" }).last().click();
    await p.getByRole("button", { name: "Run LDM Analysis" }).waitFor();
    await p.screenshot({ path: "/tmp/final-" + width + "-home.png", fullPage: true });
    assert.deepEqual(errors, []);
    await p.close();
  }
  console.log(count + " static APK-bundle interaction checks passed.");
  await b.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
