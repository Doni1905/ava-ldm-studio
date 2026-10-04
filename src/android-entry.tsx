import { createRoot } from "react-dom/client";
import { RouterProvider } from "@tanstack/react-router";
import { createHashHistory } from "@tanstack/react-router";
import { getRouter } from "./router";
import "./styles.css";
const router = getRouter();
router.update({ history: createHashHistory(), context: router.options.context });
createRoot(document.getElementById("root")!).render(<RouterProvider router={router} />);
