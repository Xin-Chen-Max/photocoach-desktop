# Contributing

Report reproducible problems through GitHub Issues. Include the application version, Windows version, display scaling, steps to reproduce, and the observed result. Do not attach API keys, private settings, project files with private paths, or photographs you cannot share.

For code changes, create a branch, keep the change focused, and run the offline suite described in [testing](docs/testing.md). Include the test result and any checks you could not run in the pull request.

Image-processing changes should include a small synthetic example with an expected result. State/history changes should cover undo/redo and project recovery. Request changes should use fake transports and cover cancellation and malformed responses.

Document changes to processing order, project schemas, user-visible behavior, and transmitted data. Keep sample paths portable. Do not add real camera files, credentials, generated verification outputs, or local environment directories to the repository.

Documentation and code comments use English. Keep comments focused on the reason for a choice or a contract that is easy to miss from the code.
