.. _faqs:

FAQs and Troubleshooting
========================

I am having trouble connecting to DataQuery using the API
---------------------------------------------------------

For the most common issues, such as incorrect credentials, invalid
certificates etc, the package will raise an exception with a helpful
error message.

If you find that the package raises an
``HTTPConnection``/``HTTPSConnectionPool`` error, please check your
proxy settings. In scenarios where an error is raised while running
``check_connection()`` (or another download), the error is raised with
context to the OAuth token request (to
“https://authe.jpmchase.com/as/token.oauth2”).

You would most likely need to pass your proxy settings to the
``JPMaQSDownload`` object, as shown in the `Connecting via a proxy
server <#connecting-via-a-proxy-server>`__ section. If you are accessing
DataQuery from an institutional/enterprise network, please contact your
IT department to ensure that you have the correct proxy settings.

For organizations using ZScaler - you may have to manually add the
ZScaler certificates to the ``certifi`` certificate store (typically
called ``cacert.pem``). You can find the location of the ``certifi``
certificate store by running the following in your Python environment:

.. code:: python

   import certifi
   print(certifi.where())

Here's a link to `ZScaler's official documentation and
FAQs <https://help.zscaler.com/zia/adding-custom-certificate-application-specific-trust-store>`__
on how to add certificates to application specific trust stores.

-  https://help.zscaler.com/zia/adding-custom-certificate-application-specific-trust-store

A function is not working as expected
-------------------------------------

Please check the documentation for the function on our `documentation
website <https://docs.macrosynergy.com>`__, and ensure you are using the
latest version of the package. If you are still having issues, please
`raise an
issue <https://github.com/macrosynergy/macrosynergy/issues/new/choose>`__
on our GitHub repository. Please include a minimal reproducible example,
and the output of ``pip freeze`` in your issue.

I have a feature request
------------------------

Please `raise an
issue <https://github.com/macrosynergy/macrosynergy/issues/new/choose>`__,
and title it “Feature Request: [your feature request]”.

Contributing or creating a pull request
---------------------------------------

The ``macrosynergy`` package is open source, but not open to external
contributions. We do not accept pull requests from users outside of the
Macrosynergy team, and any such pull requests raised against our
repository will be closed.

If you have found a bug, please `raise an
issue <https://github.com/macrosynergy/macrosynergy/issues/new/choose>`__.

If you also have a fix for it:

1. Create a public fork of the repository.
2. Make your fix in a pull request within your fork (i.e. targeting
   your fork's own branch, not our repository).
3. Raise an issue on our repository describing the problem, and include
   a link to the pull request in your fork.

The Macrosynergy team will review your fix and, where appropriate,
implement it in the package.